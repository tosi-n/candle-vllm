use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use super::block_engine::LogicalTokenBlock;
use crate::openai::sampling_params::{Logprobs, SamplingParams};
use crate::openai::streaming::ChatResponse;
use crate::tools::stream_parser::StreamToolParser;
use crate::tools::ToolCall;
use flume::Sender;
use parking_lot::Mutex;
use std::time::SystemTime;
use tokio::sync::Notify;
#[derive(Clone, PartialEq)]
pub enum SequenceStatus {
    FinishedIgnored,
    Waiting,
    Running,
    Swapped,
    Pending,
    FinishedAborted,
    Finished(String),
}

#[derive(Clone, PartialEq, Debug)]
pub enum ToolCallState {
    Normal,
    MaybeToolCall,
    InToolCall,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractiveSessionEvent {
    PrefillReady,
    Token(u32),
    Finished,
    Error(String),
}

#[derive(Debug, Default)]
struct InteractiveSessionState {
    decode_budget: usize,
    import_only: bool,
    events: VecDeque<InteractiveSessionEvent>,
}

#[derive(Debug)]
pub struct InteractiveSessionControl {
    state: Mutex<InteractiveSessionState>,
    notify: Notify,
}

impl InteractiveSessionControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(InteractiveSessionState::default()),
            notify: Notify::new(),
        })
    }

    pub fn new_import_only() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(InteractiveSessionState {
                import_only: true,
                ..InteractiveSessionState::default()
            }),
            notify: Notify::new(),
        })
    }

    pub fn decode_budget(&self) -> usize {
        self.state.lock().decode_budget
    }

    pub fn grant_decode_steps(&self, steps: usize) {
        if steps == 0 {
            return;
        }
        let mut state = self.state.lock();
        state.decode_budget = state.decode_budget.saturating_add(steps);
        self.notify.notify_waiters();
    }

    pub fn consume_decode_step(&self) {
        let mut state = self.state.lock();
        if state.decode_budget > 0 {
            state.decode_budget -= 1;
        }
    }

    pub fn is_import_only(&self) -> bool {
        self.state.lock().import_only
    }

    pub fn clear_import_only(&self) {
        self.state.lock().import_only = false;
    }

    pub fn publish_prefill_ready(&self) {
        self.state
            .lock()
            .events
            .push_back(InteractiveSessionEvent::PrefillReady);
        self.notify.notify_waiters();
    }

    pub fn publish_token(&self, token: u32) {
        self.state
            .lock()
            .events
            .push_back(InteractiveSessionEvent::Token(token));
        self.notify.notify_waiters();
    }

    pub fn publish_finished(&self) {
        self.state
            .lock()
            .events
            .push_back(InteractiveSessionEvent::Finished);
        self.notify.notify_waiters();
    }

    pub fn publish_error(&self, message: impl Into<String>) {
        self.state
            .lock()
            .events
            .push_back(InteractiveSessionEvent::Error(message.into()));
        self.notify.notify_waiters();
    }

    pub async fn next_event(&self) -> InteractiveSessionEvent {
        loop {
            if let Some(event) = self.state.lock().events.pop_front() {
                return event;
            }
            self.notify.notified().await;
        }
    }
}

pub struct SequenceData {
    prompt_token_ids: Vec<u32>,
    output_token_ids: Vec<Logprobs>,
    cumulative_logprob: f32,
    status: SequenceStatus,
    num_cached_tokens: usize, //used for chunked prefill and context cache
    // Tool call and reasoning tracking
    pub accumulated_output: String,
    pub tool_call_state: ToolCallState,
    pub tool_call_buffer: String,
    pub active_reasoning_end: Option<String>,
    pub in_code_block: bool,
    pub stream_tool_parser: Option<StreamToolParser>,
    pub pending_tool_calls: Vec<ToolCall>,
}

impl SequenceData {
    pub fn new(prompt_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.,
            status: SequenceStatus::Waiting,
            num_cached_tokens: 0,
            accumulated_output: String::new(),
            tool_call_state: ToolCallState::Normal,
            tool_call_buffer: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
            stream_tool_parser: None,
            pending_tool_calls: Vec::new(),
        }
    }

    pub fn append_token_id(&mut self, logprobs: Logprobs) {
        self.cumulative_logprob += logprobs.logprob;
        self.output_token_ids.push(logprobs);
    }

    pub fn set_status(&mut self, status: SequenceStatus) {
        self.status = status;
    }

    fn get_cumulative_logprob(&self) -> f32 {
        self.cumulative_logprob
    }
}

/// A Sequence holds information about the data it contains (the tokens), and the logical token blocks
/// to which it is mapped.
pub struct _Sequence {
    data: RwLock<SequenceData>,
    seq_id: usize,
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
}

impl _Sequence {
    pub fn new(prompt_token_ids: &Vec<u32>, seq_id: usize, block_size: usize) -> Self {
        let mut this = Self {
            data: RwLock::new(SequenceData::new(prompt_token_ids.clone())),
            seq_id,
            logical_token_blocks: Vec::new(),
            block_size,
        };
        this.append_tokens_to_blocks(prompt_token_ids);
        this
    }

    pub fn add_token(&mut self, logprobs: Logprobs) {
        self.append_token_to_blocks(logprobs.token);
        self.deref_mut().append_token_id(logprobs);
    }

    pub fn blocks_to_add_new_tok(&self) -> usize {
        let last = self.logical_token_blocks.last();
        if !last.is_some_and(|last| last.is_full() || last.is_empty()) {
            // If we have space
            0
        } else {
            1
        }
    }

    pub fn get_logical_token_blocks(&self) -> usize {
        self.logical_token_blocks.len()
    }

    pub fn get_id(&self) -> usize {
        self.seq_id
    }

    pub fn is_prompt(&self) -> bool {
        self.deref().output_token_ids.is_empty()
    }

    pub fn needs_prefill(&self) -> bool {
        self.deref().num_cached_tokens < self.deref().prompt_token_ids.len()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.deref().prompt_token_ids.len()
    }

    pub fn get_len(&self) -> usize {
        let dref = self.deref();
        dref.prompt_token_ids.len() + dref.output_token_ids.len()
    }

    pub fn get_output_len(&self) -> usize {
        self.deref().output_token_ids.len()
    }

    pub fn get_token_ids(&self) -> Vec<u32> {
        let mut res = self.deref().prompt_token_ids.clone();
        res.extend(
            self.deref()
                .output_token_ids
                .iter()
                .map(|logprobs| logprobs.token)
                .clone(),
        );
        res
    }

    pub fn get_last_token_id(&self) -> u32 {
        if self.deref().output_token_ids.is_empty() {
            *self.deref().prompt_token_ids.last().unwrap()
        } else {
            self.deref().output_token_ids.last().unwrap().token
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.deref().status,
            SequenceStatus::FinishedAborted
                | SequenceStatus::FinishedIgnored
                | SequenceStatus::Finished(_)
        )
    }

    pub fn get_status(&self) -> SequenceStatus {
        self.deref().status.clone()
    }

    pub fn get_cumulative_logprob(&self) -> f32 {
        self.deref().get_cumulative_logprob()
    }

    pub fn set_finish_reason(&mut self, finish_reason: String) {
        self.deref_mut()
            .set_status(SequenceStatus::Finished(finish_reason.clone()));
    }

    pub fn get_finish_reason(&self) -> String {
        match &self.deref().status {
            SequenceStatus::Finished(state) => state.clone(),
            SequenceStatus::FinishedAborted => "abort".to_string(),
            SequenceStatus::FinishedIgnored => "length".to_string(),
            _ => {
                unreachable!("No finish reason.")
            }
        }
    }

    #[must_use]
    /// Clones the internal logprobs.
    pub fn get_output_tokens(&self) -> Vec<Logprobs> {
        self.deref().output_token_ids.clone() // TODO(EricLBuehler): Better way to do this?
    }

    fn append_tokens_to_blocks(&mut self, tokens: &Vec<u32>) {
        for tok in tokens {
            self.append_token_to_blocks(*tok);
        }
    }

    fn append_token_to_blocks(&mut self, token: u32) {
        let last = self.logical_token_blocks.last_mut();
        match last {
            Some(last) => {
                last.append_token_id(token);
            }
            _ => {
                self.logical_token_blocks
                    .push(LogicalTokenBlock::new(self.block_size));
                self.logical_token_blocks
                    .last_mut()
                    .unwrap()
                    .append_token_id(token);
            }
        }
        if self.logical_token_blocks.last().as_ref().unwrap().is_full() {
            self.logical_token_blocks
                .push(LogicalTokenBlock::new(self.block_size));
        }
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        self.deref().num_cached_tokens
    }

    pub fn set_num_cached_tokens(&mut self, num_cached_tokens: usize) {
        self.deref_mut().num_cached_tokens = num_cached_tokens;
    }
}

impl _Sequence {
    pub fn deref(&self) -> RwLockReadGuard<'_, SequenceData> {
        // loop {
        //     if let Ok(res) = self.data.try_lock() {
        //         return res;
        //     }
        // }
        self.data.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn deref_mut(&self) -> RwLockWriteGuard<'_, SequenceData> {
        // loop {
        //     if let Ok(res) = self.data.try_lock() {
        //         return res;
        //     }
        // }
        self.data.write().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct Sequence(pub RwLock<_Sequence>);

impl Sequence {
    pub fn deref(&self) -> RwLockReadGuard<'_, _Sequence> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn deref_mut(&self) -> RwLockWriteGuard<'_, _Sequence> {
        // loop {
        //     if let Ok(v) = self.0.try_lock() {
        //         return v;
        //     }
        // }
        self.0.write().unwrap_or_else(|e| e.into_inner())
    }
}

type SeqID = usize;

/// A SequenceGroup holds the `n` (see SamplingParams) sequences generated from a single prompt.
/// A SequenceGroup contains only sequences with the same prompt. They will always be scheduled together.
pub struct SequenceGroup {
    seqs: HashMap<SeqID, Arc<Sequence>>,
    pub arrival_time: u64,
    pub group_id: usize,
    pub request_id: String,
    pub created_time: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
    pub is_embedding: bool,
    pub encoding_format: crate::openai::requests::EncodingFormat,
    pub embedding_type: crate::openai::requests::EmbeddingType,
    pub adapter_id: Option<String>,
    pub adapter_schedule: Option<Vec<crate::openai::requests::AdapterScheduleStep>>,
    pub sender: Option<Sender<ChatResponse>>,
    pub interactive_control: Option<Arc<InteractiveSessionControl>>,
    // Tool call and reasoning tracking
    pub accumulated_output: String,
    pub tool_call_state: ToolCallState,
    pub tool_call_buffer: String,
    pub active_reasoning_end: Option<String>,
    pub in_code_block: bool,
}

impl SequenceGroup {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seqs: &[Arc<Sequence>],
        arrival_time: u64,
        group_id: usize,
        request_id: String,
        created_time: SystemTime,
        sampling_params: SamplingParams,
        use_logprobs: bool,
        is_embedding: bool,
        encoding_format: crate::openai::requests::EncodingFormat,
        embedding_type: crate::openai::requests::EmbeddingType,
        adapter_id: Option<String>,
        adapter_schedule: Option<Vec<crate::openai::requests::AdapterScheduleStep>>,
        interactive_control: Option<Arc<InteractiveSessionControl>>,
        sender: Option<Sender<ChatResponse>>,
    ) -> Self {
        let mut seq_map = HashMap::new();
        for seq in seqs {
            seq_map.insert(seq.deref_mut().get_id(), seq.clone());
        }
        Self {
            seqs: seq_map,
            arrival_time,
            group_id,
            request_id,
            created_time,
            sampling_params,
            use_logprobs,
            is_embedding,
            encoding_format,
            embedding_type,
            adapter_id,
            adapter_schedule,
            sender,
            interactive_control,
            accumulated_output: "".to_string(),
            tool_call_state: ToolCallState::Normal,
            tool_call_buffer: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
        }
    }

    pub fn set_status(&self, status: SequenceStatus) {
        // for seq in self.seqs.values() {
        //     seq.deref_mut().deref().set_status(status.clone());
        // }
        for seq in self.seqs.values() {
            // Lock each sequence individually and set the status
            if let Ok(seq_guard) = seq.0.write() {
                seq_guard.deref_mut().set_status(status.clone());
            }
        }
    }

    pub fn get_status(&self) -> SequenceStatus {
        self.seqs.values().nth(0).unwrap().deref().get_status()
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref().blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.seqs.len()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref().get_logical_token_blocks())
            .sum()
    }

    pub fn get_seqs(&self) -> &HashMap<SeqID, Arc<Sequence>> {
        &self.seqs
    }

    pub fn arrival_time(&self) -> u64 {
        self.arrival_time
    }

    pub fn get_id(&self) -> &usize {
        &self.group_id
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|(_, x)| x.deref().is_finished())
    }

    pub fn get_request_id(&self) -> &String {
        &self.request_id
    }

    pub fn get_created_time(&self) -> SystemTime {
        self.created_time
    }

    pub fn adapter_id(&self) -> Option<&str> {
        self.adapter_id.as_deref()
    }

    pub fn interactive_control(&self) -> Option<Arc<InteractiveSessionControl>> {
        self.interactive_control.clone()
    }

    pub fn is_decode_blocked(&self) -> bool {
        let Some(control) = self.interactive_control.as_ref() else {
            return false;
        };
        let Some(seq) = self.seqs.values().next() else {
            return false;
        };
        !seq.deref().needs_prefill() && control.decode_budget() == 0
    }

    pub fn resolve_decode_adapter_id(&self) -> Option<String> {
        let Some(timeline) = self.adapter_schedule.as_ref() else {
            return self.adapter_id.clone();
        };
        if timeline.is_empty() {
            return self.adapter_id.clone();
        }

        let step = self
            .seqs
            .values()
            .next()
            .map(|seq| seq.deref().get_output_len())
            .unwrap_or(0);

        let mut selected = self.adapter_id.clone();
        for event in timeline {
            if event.start_step <= step {
                selected = Some(event.adapter_id.clone());
            } else {
                break;
            }
        }
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::sampling_params::{EarlyStoppingCondition, SamplingParams};
    use tokio::runtime::Runtime;

    #[test]
    fn resolve_decode_adapter_id_switches_at_step_boundaries() {
        let seq = Arc::new(Sequence(std::sync::RwLock::new(_Sequence::new(
            &vec![1, 2, 3],
            42,
            16,
        ))));
        let sampling_params = SamplingParams::new(
            1,
            None,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            false,
            1.0,
            EarlyStoppingCondition::UnlikelyBetterCandidates,
            None,
            vec![],
            false,
            16,
            None,
            None,
            true,
            None,
        )
        .expect("sampling params");
        let group = SequenceGroup::new(
            &[seq.clone()],
            0,
            1,
            "req-1".to_string(),
            SystemTime::now(),
            sampling_params,
            false,
            false,
            crate::openai::requests::EncodingFormat::Float,
            crate::openai::requests::EmbeddingType::Last,
            Some("base".to_string()),
            Some(vec![
                crate::openai::requests::AdapterScheduleStep {
                    start_step: 0,
                    adapter_id: "planner".to_string(),
                },
                crate::openai::requests::AdapterScheduleStep {
                    start_step: 2,
                    adapter_id: "verifier".to_string(),
                },
            ]),
            None,
            None,
        );

        assert_eq!(
            group.resolve_decode_adapter_id().as_deref(),
            Some("planner")
        );

        seq.deref_mut()
            .add_token(crate::openai::sampling_params::Logprobs {
                token: 5,
                logprob: 0.0,
                bytes: "a".to_string(),
                top_logprobs: Vec::new(),
            });
        assert_eq!(
            group.resolve_decode_adapter_id().as_deref(),
            Some("planner")
        );

        seq.deref_mut()
            .add_token(crate::openai::sampling_params::Logprobs {
                token: 6,
                logprob: 0.0,
                bytes: "b".to_string(),
                top_logprobs: Vec::new(),
            });
        assert_eq!(
            group.resolve_decode_adapter_id().as_deref(),
            Some("verifier")
        );
    }

    #[test]
    fn interactive_control_queues_events_and_budget() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let control = InteractiveSessionControl::new_import_only();
            assert!(control.is_import_only());
            assert_eq!(control.decode_budget(), 0);

            control.publish_prefill_ready();
            control.grant_decode_steps(2);
            control.consume_decode_step();
            control.publish_token(42);
            control.publish_finished();

            assert_eq!(control.decode_budget(), 1);
            assert_eq!(
                control.next_event().await,
                InteractiveSessionEvent::PrefillReady
            );
            assert_eq!(
                control.next_event().await,
                InteractiveSessionEvent::Token(42)
            );
            assert_eq!(
                control.next_event().await,
                InteractiveSessionEvent::Finished
            );
        });
    }
}
