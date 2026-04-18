package com.brahmadeo.tunedllm

enum class Role {
    USER, ASSISTANT
}

data class ChatMessage(
    val id: String,
    val role: Role,
    val content: String
)

data class ChatSession(
    val id: String,
    val title: String,
    val messages: List<ChatMessage> = emptyList(),
    val systemPrompt: String? = null
)

data class GenerationConfig(
    val temperature: Float = 0.7f,
    val topP: Float = 0.8f,
    val topK: Int = 20,
    val minP: Float = 0.05f,
    val presencePenalty: Float = 1.5f,
    val repetitionPenalty: Float = 1.1f,
    val enableThinking: Boolean = true,
    val mode: GenerationMode = GenerationMode.GENERAL
)

enum class GenerationMode {
    GENERAL, CODING, REASONING
}

data class ChatTemplate(
    val prefix: String,
    val roleSuffix: String,
    val eot: String,
    val stopStrings: List<String>,
    val userRole: String,
    val assistantRole: String,
    val thinkStartTag: String? = null,
    val thinkEndTag: String? = null,
    val shouldPruneThinkingFromHistory: Boolean = false
) {
    companion object {
        val GEMMA = ChatTemplate(
            prefix = "<start_of_turn>",
            roleSuffix = "\n",
            eot = "<end_of_turn>\n",
            stopStrings = listOf("<end_of_turn>", "<eos>", "</s>"),
            userRole = "user",
            assistantRole = "model"
        )

        val QWEN = ChatTemplate(
            prefix = "<|im_start|>",
            roleSuffix = "\n",
            eot = "<|im_end|>\n",
            stopStrings = listOf("<|im_end|>", "<|endoftext|>", "<|im_start|>"),
            userRole = "user",
            assistantRole = "assistant",
            thinkStartTag = "<think>",
            thinkEndTag = "</think>",
            shouldPruneThinkingFromHistory = true
        )

        val GEMMA4 = ChatTemplate(
            prefix = "<|turn>",
            roleSuffix = "\n",
            eot = "<turn|>\n",
            stopStrings = listOf("<turn|>", "<eos>", "</s>"),
            userRole = "user",
            assistantRole = "model",
            thinkStartTag = "<|channel>thought",
            thinkEndTag = "<channel|>",
            shouldPruneThinkingFromHistory = true
        )

        val LLAMA3 = ChatTemplate(
            prefix = "<|start_header_id|>",
            roleSuffix = "<|end_header_id|>\n\n",
            eot = "<|eot_id|>",
            stopStrings = listOf("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"),
            userRole = "user",
            assistantRole = "assistant"
        )

        fun fromModelName(name: String?): ChatTemplate {
            val n = name?.lowercase() ?: ""
            return when {
                n.contains("qwen") -> QWEN
                n.contains("gemma-4") || n.contains("gemma4") -> GEMMA4
                n.contains("llama-3") || n.contains("llama3") -> LLAMA3
                else -> GEMMA
            }
        }
    }
}

data class ChatState(
    val sessions: List<ChatSession> = emptyList(),
    val currentSessionId: String? = null,
    val isGenerating: Boolean = false,
    val currentTps: Float? = null,
    val currentStatus: String? = null,
    val lastPpStatus: String? = null,
    val lastTgStatus: String? = null,
    val lastGenerationStatus: String? = null,
    val isModelLoaded: Boolean = false,
    val isAutoLoading: Boolean = false,
    val lastModelPath: String? = null,
    val modelName: String? = null,
    val isCopying: Boolean = false,
    val copyProgress: Float = 0f,
    val chatTemplate: ChatTemplate = ChatTemplate.GEMMA,
    val config: GenerationConfig = GenerationConfig(),
    val error: String? = null
) {
    val messages: List<ChatMessage>
        get() = sessions.find { it.id == currentSessionId }?.messages ?: emptyList()
}
