package com.brahmadeo.tunedllm

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import dev.jeziellago.compose.markdowntext.MarkdownText
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.text.font.FontStyle

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: LlmViewModel, onModelPicker: () -> Unit) {
    val state by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }
    var showTemplateDialog by remember { mutableStateOf(false) }
    var showProfileMenu by remember { mutableStateOf(false) }

    val currentView = LocalView.current
    DisposableEffect(state.isGenerating) {
        if (state.isGenerating) {
            currentView.keepScreenOn = true
        }
        onDispose {
            currentView.keepScreenOn = false
        }
    }

    val isAtBottom by remember {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val visibleItemsInfo = layoutInfo.visibleItemsInfo
            if (layoutInfo.totalItemsCount == 0) {
                true
            } else {
                val lastVisibleItem = visibleItemsInfo.lastOrNull()
                val isLastItemVisible = lastVisibleItem?.let { it.index == layoutInfo.totalItemsCount - 1 } ?: false
                if (!isLastItemVisible) return@derivedStateOf false
                
                // Also check if the bottom of the last item is actually visible
                val lastItemBottom = (lastVisibleItem?.offset ?: 0) + (lastVisibleItem?.size ?: 0)
                val viewportBottom = layoutInfo.viewportEndOffset
                lastItemBottom <= viewportBottom + 100 // small buffer
            }
        }
    }

    LaunchedEffect(state.messages.lastOrNull()?.content) {
        // Only auto-scroll if we are currently generating AND we were already at the bottom
        if (state.isGenerating && isAtBottom) {
            if (state.messages.isNotEmpty()) {
                // Scroll to the last item with a large offset to ensure we hit the bottom of the bubble
                listState.animateScrollToItem(state.messages.size - 1, 10000)
            }
        }
    }

    if (showTemplateDialog) {
        AlertDialog(
            onDismissRequest = { showTemplateDialog = false },
            title = { Text("Chat Template Info") },
            text = {
                Column {
                    Text("Current model: ${state.modelName ?: "Default"}")
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("User Role: ${state.chatTemplate.userRole}")
                    Text("Assistant Role: ${state.chatTemplate.assistantRole}")
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("Stop Tokens: ${state.chatTemplate.stopStrings.joinToString(", ")}", style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = {
                TextButton(onClick = { showTemplateDialog = false }) {
                    Text("Close")
                }
            }
        )
    }

    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                Column(modifier = Modifier.fillMaxSize()) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Button(
                        onClick = { 
                            viewModel.createNewSession()
                            scope.launch { drawerState.close() }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("New Chat")
                    }
                    Divider(modifier = Modifier.padding(vertical = 8.dp))
                    
                    LazyColumn(modifier = Modifier.weight(1f)) {
                        items(state.sessions) { session ->
                            NavigationDrawerItem(
                                label = { 
                                    Text(
                                        text = session.title,
                                        maxLines = 1,
                                        modifier = Modifier.weight(1f)
                                    )
                                },
                                selected = session.id == state.currentSessionId,
                                onClick = {
                                    viewModel.selectSession(session.id)
                                    scope.launch { drawerState.close() }
                                },
                                badge = {
                                    IconButton(onClick = { viewModel.deleteSession(session.id) }) {
                                        Icon(Icons.Default.Delete, contentDescription = "Delete", tint = MaterialTheme.colorScheme.error)
                                    }
                                },
                                modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                            )
                        }
                    }

                    Divider()
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Model: ${state.modelName ?: state.lastModelPath?.substringAfterLast("/") ?: "None"}",
                            style = MaterialTheme.typography.labelMedium,
                            maxLines = 1
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        OutlinedButton(
                            onClick = { 
                                viewModel.unloadModel()
                                onModelPicker()
                                scope.launch { drawerState.close() }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Change Model")
                        }
                    }
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { 
                        val currentTitle = state.sessions.find { it.id == state.currentSessionId }?.title ?: "Tuned LLM Chat"
                        Text(currentTitle, maxLines = 1) 
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Menu")
                        }
                    },
                    actions = {
                        Box {
                            IconButton(onClick = { showProfileMenu = true }) {
                                val icon = when (state.config.mode) {
                                    GenerationMode.GENERAL -> Icons.Default.Chat
                                    GenerationMode.CODING -> Icons.Default.Code
                                    GenerationMode.REASONING -> Icons.Default.AutoFixHigh
                                }
                                Icon(icon, contentDescription = "Profile")
                            }
                            DropdownMenu(
                                expanded = showProfileMenu,
                                onDismissRequest = { showProfileMenu = false }
                            ) {
                                DropdownMenuItem(
                                    text = { Text("General Profile") },
                                    leadingIcon = { Icon(Icons.Default.Chat, null) },
                                    onClick = { viewModel.updateGenerationMode(GenerationMode.GENERAL); showProfileMenu = false }
                                )
                                DropdownMenuItem(
                                    text = { Text("Coding Profile") },
                                    leadingIcon = { Icon(Icons.Default.Code, null) },
                                    onClick = { viewModel.updateGenerationMode(GenerationMode.CODING); showProfileMenu = false }
                                )
                                DropdownMenuItem(
                                    text = { Text("Reasoning Profile") },
                                    leadingIcon = { Icon(Icons.Default.AutoFixHigh, null) },
                                    onClick = { viewModel.updateGenerationMode(GenerationMode.REASONING); showProfileMenu = false }
                                )
                            }
                        }
                        IconButton(onClick = { 
                            val chatText = state.messages.joinToString("\n\n") { 
                                "${if (it.role == Role.USER) "USER" else "ASSISTANT"}: ${it.content}" 
                            }
                            viewModel.copyToClipboard(chatText)
                        }) {
                            Icon(Icons.Default.ContentCopy, contentDescription = "Copy Chat")
                        }
                        IconButton(onClick = { showTemplateDialog = true }) {
                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        titleContentColor = MaterialTheme.colorScheme.primary
                    )
                )
            },
            bottomBar = {
                Surface(tonalElevation = 8.dp) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                    ) {
                        if (state.error != null) {
                            Text(
                                text = "Error: ${state.error}",
                                color = MaterialTheme.colorScheme.error,
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                        }

                        // Options Row
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Checkbox(
                                    checked = state.config.enableThinking,
                                    onCheckedChange = { viewModel.toggleThinking(it) }
                                )
                                Text("Thinking Mode", style = MaterialTheme.typography.labelMedium)
                            }
                            
                            if (state.isGenerating) {
                                Text(
                                    text = state.currentStatus ?: "Processing...",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            } else if (state.lastGenerationStatus != null) {
                                Text(
                                    text = state.lastGenerationStatus!!,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.outline
                                )
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            TextField(
                                value = inputText,
                                onValueChange = { inputText = it },
                                modifier = Modifier.weight(1f),
                                placeholder = { Text("Enter prompt...") },
                                maxLines = 4,
                                shape = RoundedCornerShape(24.dp),
                                colors = TextFieldDefaults.colors(
                                    focusedIndicatorColor = Color.Transparent,
                                    unfocusedIndicatorColor = Color.Transparent
                                )
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Button(
                                onClick = {
                                    if (state.isGenerating) {
                                        viewModel.stopGeneration()
                                    } else {
                                        if (inputText.isNotBlank()) {
                                            viewModel.generate(inputText)
                                            inputText = ""
                                        }
                                    }
                                },
                                enabled = state.isModelLoaded,
                                shape = RoundedCornerShape(24.dp),
                                contentPadding = PaddingValues(12.dp)
                            ) {
                                Icon(if (state.isGenerating) Icons.Default.Stop else Icons.Default.Send, null)
                            }
                        }
                    }
                }
            }
        ) { paddingValues ->
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(horizontal = 8.dp),
                contentPadding = PaddingValues(vertical = 8.dp)
            ) {
                items(state.messages, key = { it.id }) { message ->
                    ChatMessageItem(message, viewModel)
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
fun ChatMessageItem(message: ChatMessage, viewModel: LlmViewModel) {
    val isUser = message.role == Role.USER
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val containerColor = if (isUser) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.secondaryContainer
    }
    val contentColor = if (isUser) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSecondaryContainer
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            color = containerColor,
            contentColor = contentColor,
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 0.dp,
                bottomEnd = if (isUser) 0.dp else 16.dp
            ),
            tonalElevation = 2.dp
        ) {
            Column {
                MarkdownContent(message.content, contentColor, viewModel)
                
                Box(
                    modifier = Modifier.fillMaxWidth().padding(4.dp),
                    contentAlignment = Alignment.BottomEnd
                ) {
                    IconButton(
                        onClick = { viewModel.copyToClipboard(message.content) },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            Icons.Default.ContentCopy, 
                            contentDescription = "Copy Message",
                            tint = contentColor.copy(alpha = 0.6f),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }
        Text(
            text = if (isUser) "You" else "Assistant",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.outline,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
        )
    }
}

@Composable
fun MarkdownContent(content: String, textColor: Color, viewModel: LlmViewModel) {
    val state by viewModel.uiState.collectAsState()
    val template = state.chatTemplate
    
    // Split by model-specific think tags
    val thinkParts = remember(content, template) {
        val startTag = template.thinkStartTag ?: "<think>"
        val endTag = template.thinkEndTag ?: "</think>"
        
        // Escape for regex
        val eStart = Regex.escape(startTag)
        val eEnd = Regex.escape(endTag)
        
        val regex = Regex("($eStart[\\s\\S]*?$eEnd|$eStart[\\s\\S]*$)")
        val matches = regex.findAll(content)
        val result = mutableListOf<Pair<String, Boolean>>() // text to isThinking
        
        var lastIndex = 0
        for (match in matches) {
            if (match.range.first > lastIndex) {
                result.add(content.substring(lastIndex, match.range.first) to false)
            }
            result.add(match.value to true)
            lastIndex = match.range.last + 1
        }
        if (lastIndex < content.length) {
            result.add(content.substring(lastIndex) to false)
        }
        if (result.isEmpty()) result.add(content to false)
        result
    }

    Column(modifier = Modifier.padding(12.dp)) {
        SelectionContainer {
            Column {
                thinkParts.forEach { (partText, isThinking) ->
                    if (isThinking) {
                        // Styled thinking block
                        Surface(
                            color = textColor.copy(alpha = 0.05f),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.padding(vertical = 4.dp).fillMaxWidth()
                        ) {
                            val cleanThink = partText
                                .replace(template.thinkStartTag ?: "<think>", "")
                                .replace(template.thinkEndTag ?: "</think>", "")
                                .trim()
                            
                            Column(modifier = Modifier.padding(8.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.AutoFixHigh, null, modifier = Modifier.size(14.dp), tint = textColor.copy(alpha = 0.5f))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Reasoning", style = MaterialTheme.typography.labelSmall, color = textColor.copy(alpha = 0.5f))
                                }
                                if (cleanThink.isNotEmpty()) {
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = cleanThink,
                                        style = MaterialTheme.typography.bodySmall.copy(
                                            color = textColor.copy(alpha = 0.7f),
                                            fontStyle = FontStyle.Italic
                                        )
                                    )
                                } else {
                                    // Show a placeholder while thinking is starting
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Thinking...",
                                        style = MaterialTheme.typography.bodySmall.copy(
                                            color = textColor.copy(alpha = 0.3f),
                                            fontStyle = FontStyle.Italic
                                        )
                                    )
                                }
                            }
                        }
                    } else {
                        // Standard markdown part (handle code blocks within here)
                        RenderStandardContent(partText, textColor, viewModel)
                    }
                }
            }
        }
    }
}

@Composable
fun RenderStandardContent(content: String, textColor: Color, viewModel: LlmViewModel) {
    val parts = remember(content) {
        val regex = Regex("```[a-zA-Z]*\\n?([\\s\\S]*?)```")
        val matches = regex.findAll(content)
        val result = mutableListOf<Pair<String, Boolean>>() // text to isCodeBlock
        
        var lastIndex = 0
        for (match in matches) {
            if (match.range.first > lastIndex) {
                result.add(content.substring(lastIndex, match.range.first) to false)
            }
            result.add(match.groupValues[1] to true)
            lastIndex = match.range.last + 1
        }
        if (lastIndex < content.length) {
            result.add(content.substring(lastIndex) to false)
        }
        if (result.isEmpty()) result.add(content to false)
        result
    }

    parts.forEach { (text, isCode) ->
        if (isCode) {
            Surface(
                color = Color.Black.copy(alpha = 0.05f),
                shape = RoundedCornerShape(4.dp),
                modifier = Modifier.padding(vertical = 4.dp).fillMaxWidth()
            ) {
                Column {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                        contentAlignment = Alignment.CenterEnd
                    ) {
                        TextButton(
                            onClick = { viewModel.copyToClipboard(text) },
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                            modifier = Modifier.height(24.dp)
                        ) {
                            Icon(Icons.Default.ContentCopy, null, modifier = Modifier.size(14.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Copy", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    Text(
                        text = text.trim(),
                        style = MaterialTheme.typography.bodySmall.copy(color = textColor),
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
                    )
                }
            }
        } else if (text.isNotBlank()) {
            // Pre-process common LaTeX symbols that the markdown library doesn't support
            val processedText = text
                .replace("$\\rightarrow$", "→").replace("\\rightarrow", "→")
                .replace("$\\leftarrow$", "←").replace("\\leftarrow", "←")
                .replace("$\\Rightarrow$", "⇒").replace("\\Rightarrow", "⇒")
                .replace("$\\Leftarrow$", "⇐").replace("\\Leftarrow", "⇐")
                .replace("$\\leftrightarrow$", "↔").replace("\\leftrightarrow", "↔")
                .replace("$\\dots$", "…").replace("\\dots", "…")

            MarkdownText(
                markdown = processedText,
                style = MaterialTheme.typography.bodyLarge.copy(color = textColor),
                syntaxHighlightColor = if (isSystemInDarkTheme()) Color(0xFF2D2D2D) else Color(0xFFF5F5F5)
            )
        }
    }
}
