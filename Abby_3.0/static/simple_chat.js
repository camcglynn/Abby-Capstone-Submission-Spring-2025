// Simple chat.js - Minimal implementation for debugging
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    
    // Set up event listeners
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message) {
                sendMessage(message);
            }
        });
    }
    
    // Enable send button only when there's text
    if (userInput) {
        userInput.addEventListener('input', function() {
            if (sendButton) {
                sendButton.disabled = !userInput.value.trim();
            }
        });
        
        // Also handle Enter key
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && userInput.value.trim()) {
                e.preventDefault();
                sendMessage(userInput.value.trim());
            }
        });
    }
    
    // Session ID management
    let sessionId = localStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Date.now();
        localStorage.setItem('chat_session_id', sessionId);
    }
    
    // Add a user message to the chat
    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'user-message';
        messageDiv.textContent = message;
        
        const container = document.createElement('div');
        container.className = 'message-container user-container';
        container.appendChild(messageDiv);
        
        if (chatMessages) {
            chatMessages.appendChild(container);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Add a bot message to the chat
    function addBotMessage(message, citations = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'bot-message';
        
        // Use textContent for safety
        const textPara = document.createElement('p');
        textPara.textContent = message;
        messageDiv.appendChild(textPara);
        
        // Add simple citation links if available
        if (citations && Array.isArray(citations) && citations.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.innerHTML = '<p><strong>Sources:</strong></p>';
            
            const sourcesList = document.createElement('ul');
            citations.forEach(citation => {
                if (citation.url) {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = citation.url;
                    a.target = '_blank';
                    a.textContent = citation.title || citation.source || 'Source';
                    li.appendChild(a);
                    sourcesList.appendChild(li);
                }
            });
            
            if (sourcesList.children.length > 0) {
                sourcesDiv.appendChild(sourcesList);
                messageDiv.appendChild(sourcesDiv);
            }
        }
        
        const container = document.createElement('div');
        container.className = 'message-container bot-container';
        container.appendChild(messageDiv);
        
        if (chatMessages) {
            chatMessages.appendChild(container);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        return messageDiv;
    }
    
    // Add typing indicator
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'bot-message typing-indicator';
        typingDiv.innerHTML = '<span>Abby is typing</span><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
        
        const container = document.createElement('div');
        container.className = 'message-container bot-container typing-container';
        container.appendChild(typingDiv);
        
        if (chatMessages) {
            chatMessages.appendChild(container);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const typingContainer = document.querySelector('.typing-container');
        if (typingContainer) {
            typingContainer.remove();
        }
    }
    
    // Send a message to the server
    function sendMessage(message) {
        // Don't send empty messages
        if (!message || !message.trim()) {
            return;
        }
        
        // Add user message to UI
        addUserMessage(message);
        
        // Show typing indicator
        addTypingIndicator();
        
        // Clear input field
        if (userInput) {
            userInput.value = '';
            if (sendButton) {
                sendButton.disabled = true;
            }
        }
        
        // Prepare data for request
        const data = {
            message: message,
            session_id: sessionId
        };
        
        // Send request to server
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            console.log('Server response:', data);
            
            // Extract message text
            let messageText = '';
            if (typeof data === 'string') {
                messageText = data;
            } else if (data && typeof data === 'object') {
                messageText = data.text || data.message || data.content || 'No response text found';
            }
            
            // Get citation objects
            const citations = data.citation_objects || [];
            
            // Add bot message to UI
            addBotMessage(messageText, citations);
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Show error message
            addBotMessage(`I'm sorry, I encountered an error: ${error.message}. Please try again.`);
        });
    }
    
    // Add initial welcome message
    setTimeout(() => {
        addBotMessage('Hi! 👋 How can I help you today?');
    }, 500);
}); 