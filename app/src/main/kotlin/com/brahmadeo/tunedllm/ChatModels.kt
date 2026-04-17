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
    val messages: List<ChatMessage> = emptyList()
)

data class ChatTemplate(
    val prefix: String,
    val roleSuffix: String,
    val eot: String,
    val stopStrings: List<String>,
    val userRole: String,
    val assistantRole: String,
    val systemRole: String = "system"
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
            stopStrings = listOf("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"),
            userRole = "user",
            assistantRole = "assistant"
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
    val isModelLoaded: Boolean = false,
    val isAutoLoading: Boolean = false,
    val lastModelPath: String? = null,
    val modelName: String? = null,
    val isCopying: Boolean = false,
    val copyProgress: Float = 0f,
    val chatTemplate: ChatTemplate = ChatTemplate.GEMMA,
    val error: String? = null
) {
    val messages: List<ChatMessage>
        get() = sessions.find { it.id == currentSessionId }?.messages ?: emptyList()
}
