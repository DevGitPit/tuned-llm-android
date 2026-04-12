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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: LlmViewModel, onModelPicker: () -> Unit) {
    val state by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }
    var showTemplateDialog by remember { mutableStateOf(false) }

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
                lastVisibleItem?.index == layoutInfo.totalItemsCount - 1
            }
        }
    }

    // Improved auto-scrolling: only scroll if at bottom or when a new message starts
    LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.content) {
        if (isAtBottom || state.isGenerating) {
            if (state.messages.isNotEmpty()) {
                listState.animateScrollToItem(state.messages.size - 1)
            }
        }
    }

    if (showTemplateDialog) {
        var templateText by remember { mutableStateOf(state.chatTemplate) }
        AlertDialog(
            onDismissRequest = { showTemplateDialog = false },
            title = { Text("Edit Chat Template") },
            text = {
                Column {
                    Text("Use {{prompt}} as placeholder for user message.", style = MaterialTheme.typography.bodySmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    TextField(
                        value = templateText,
                        onValueChange = { templateText = it },
                        modifier = Modifier.fillMaxWidth(),
                        maxLines = 10
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.updateChatTemplate(templateText)
                    showTemplateDialog = false
                }) {
                    Text("Save")
                }
            },
            dismissButton = {
                TextButton(onClick = { showTemplateDialog = false }) {
                    Text("Cancel")
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
                        Text(currentTitle) 
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Menu")
                        }
                    },
                    actions = {
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
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            TextField(
                                value = inputText,
                                onValueChange = { inputText = it },
                                modifier = Modifier.weight(1f),
                                placeholder = { Text("Enter prompt...") },
                                maxLines = 4
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
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Text(if (state.isGenerating) "Stop" else "Send")
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
                topStart = 12.dp,
                topEnd = 12.dp,
                bottomStart = if (isUser) 12.dp else 0.dp,
                bottomEnd = if (isUser) 0.dp else 12.dp
            ),
            tonalElevation = 2.dp
        ) {
            Column {
                MarkdownContent(message.content, contentColor, viewModel)
                
                // Message-level copy button at the bottom of the bubble
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
            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
        )
    }
}

@Composable
fun MarkdownContent(content: String, textColor: Color, viewModel: LlmViewModel) {
    // Basic code block splitting logic
    val parts = remember(content) {
        val regex = Regex("```[a-zA-Z]*\\n?([\\s\\S]*?)```")
        val matches = regex.findAll(content)
        val result = mutableListOf<Pair<String, Boolean>>() // text to isCodeBlock
        
        var lastIndex = 0
        for (match in matches) {
            if (match.range.first > lastIndex) {
                result.add(content.substring(lastIndex, match.range.first) to false)
            }
            // Group 1 is the code content
            result.add(match.groupValues[1] to true)
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
                        MarkdownText(
                            markdown = text,
                            style = MaterialTheme.typography.bodyMedium.copy(color = textColor),
                            syntaxHighlightColor = if (isSystemInDarkTheme()) Color(0xFF2D2D2D) else Color(0xFFF5F5F5)
                        )
                    }
                }
            }
        }
    }
}
