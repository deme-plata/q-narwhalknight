//! Chat Template Formatting for Different Models
//!
//! This module provides prompt formatting utilities for different LLM models.
//! Each model family has its own instruction format that must be followed for optimal results.

use tracing::debug;

/// Format a user message with the appropriate chat template for the given model.
///
/// # Arguments
/// * `model_name` - The name of the model (e.g., "Mistral-7B-Instruct-v0.3", "Mistral-Small-3.2-24B")
/// * `user_message` - The raw user message to format
///
/// # Returns
/// A formatted prompt string ready for inference
///
/// # Examples
/// ```
/// use q_ai_inference::format_chat_prompt;
///
/// let prompt = format_chat_prompt("Mistral-7B-Instruct-v0.3", "What is quantum computing?");
/// assert_eq!(prompt, "[INST] What is quantum computing? [/INST]");
/// ```
pub fn format_chat_prompt(model_name: &str, user_message: &str) -> String {
    // Detect model family and apply appropriate template
    if model_name.contains("Mistral-Small") || model_name.contains("Mistral-3.2") {
        // Mistral Small 3.2 format (24B parameter model)
        // Uses SYSTEM_PROMPT tags for better instruction following
        debug!("Using Mistral Small 3.2 chat template");
        format!(
            "[SYSTEM_PROMPT]You are a helpful, harmless, and honest AI assistant. Always answer as helpfully as possible, while being safe. If a question does not make sense or is not factually coherent, explain why instead of answering incorrectly.[/SYSTEM_PROMPT][INST]{}[/INST]",
            user_message
        )
    } else if model_name.contains("Mistral-7B") || model_name.contains("Mistral") {
        // Mistral 7B Instruct format (original Mistral instruction format)
        debug!("Using Mistral 7B Instruct chat template");
        format!("[INST] {} [/INST]", user_message)
    } else if model_name.contains("Llama-3") || model_name.contains("Llama3") {
        // Llama 3 format with special tokens
        debug!("Using Llama 3 chat template");
        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_message
        )
    } else if model_name.contains("Llama") {
        // Llama 2 format
        debug!("Using Llama 2 chat template");
        format!(
            "<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n{} [/INST]",
            user_message
        )
    } else if model_name.contains("Qwen") {
        // Qwen format
        debug!("Using Qwen chat template");
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_message
        )
    } else if model_name.contains("Phi") {
        // Phi-3 format
        debug!("Using Phi-3 chat template");
        format!(
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
            user_message
        )
    } else {
        // Default format - just pass through the message
        // This works for base models or models without special formatting
        debug!("Using default (no template) format");
        user_message.to_string()
    }
}

/// Format a multi-turn conversation with the appropriate chat template.
///
/// # Arguments
/// * `model_name` - The name of the model
/// * `messages` - A slice of (role, content) tuples representing the conversation
///
/// # Returns
/// A formatted prompt string ready for inference
pub fn format_conversation(model_name: &str, messages: &[(&str, &str)]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    // For now, just format the last user message
    // TODO: Implement full conversation history formatting for each model
    let last_user_message = messages
        .iter()
        .rev()
        .find(|(role, _)| *role == "user")
        .map(|(_, content)| *content)
        .unwrap_or("");

    format_chat_prompt(model_name, last_user_message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_7b_format() {
        let prompt = format_chat_prompt("Mistral-7B-Instruct-v0.3", "Hello");
        assert_eq!(prompt, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_mistral_small_format() {
        let prompt = format_chat_prompt("Mistral-Small-3.2-24B", "Hello");
        assert!(prompt.contains("[SYSTEM_PROMPT]"));
        assert!(prompt.contains("[INST]Hello[/INST]"));
    }

    #[test]
    fn test_llama_3_format() {
        let prompt = format_chat_prompt("Llama-3-8B-Instruct", "Hello");
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_default_format() {
        let prompt = format_chat_prompt("unknown-model", "Hello");
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn test_empty_conversation() {
        let messages: &[(&str, &str)] = &[];
        let prompt = format_conversation("Mistral-7B", messages);
        assert_eq!(prompt, "");
    }

    #[test]
    fn test_conversation_with_user_message() {
        let messages = &[
            ("system", "You are helpful."),
            ("user", "What is 2+2?"),
        ];
        let prompt = format_conversation("Mistral-7B", messages);
        assert_eq!(prompt, "[INST] What is 2+2? [/INST]");
    }
}
