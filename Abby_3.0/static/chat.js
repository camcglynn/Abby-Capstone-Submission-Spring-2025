// Make startNewChat function globally available immediately
window.startNewChat = function() {
    console.log("Starting new chat session");
    
    // First, force save the current chat to history if it has content
    const chatMessages = document.getElementById('chatMessages');
    if (window.conversationId && chatMessages && chatMessages.childElementCount > 1) {
        console.log("Force saving current chat to history before starting new chat:", window.conversationId);
        saveChatToHistory(window.conversationId);
    } else {
        console.log("No current chat to save or empty chat");
    }
    
    // Create a new conversation ID for this chat
    const oldConversationId = window.conversationId;
    window.conversationId = 'chat_' + Date.now();
    console.log("Created new conversation ID:", window.conversationId);
    
    // Clear any previous conversation state
    if (window.clearSessionHistory) {
        window.clearSessionHistory(true);
    }
    
    // Reset message counter
    window.messageCount = 0;
    
    // Show a loading indicator briefly
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'system-message';
    loadingIndicator.textContent = 'Starting new chat...';
    chatMessages.innerHTML = '';
    chatMessages.appendChild(loadingIndicator);
    
    // DIRECT MANIPULATION: Force a refresh of chat history display BEFORE clearing
    console.log("DEBUG: Forcing immediate history update BEFORE clearing");
    initializeChatHistory();
    
    setTimeout(() => {
        // Clear messages and show welcome
        chatMessages.innerHTML = '';
        
        // Add welcome message
        const welcomeContainer = document.createElement('div');
        welcomeContainer.className = 'message-container';
        welcomeContainer.setAttribute('role', 'listitem');
        welcomeContainer.setAttribute('aria-label', 'Abby\'s welcome message');
        
        const welcomeMessage = document.createElement('div');
        welcomeMessage.className = 'message bot-message';
        welcomeMessage.innerHTML = "<p>Hi! 👋 How can I help you today?</p>";
        
        welcomeContainer.appendChild(welcomeMessage);
        chatMessages.appendChild(welcomeContainer);
        
        // Add suggestion prompts
        addSuggestionPrompts();
        
        // Enable input
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');
        if (userInput) {
            userInput.disabled = false;
            userInput.focus();
        }
        if (sendButton) {
            sendButton.disabled = true;
        }
        
        // Update chat history to reflect the new chat
        console.log("Reinitializing chat history...");
        
        // DIRECT MANIPULATION: Force another refresh after updating
        console.log("DEBUG: Forcing another immediate history update AFTER clearing");
        initializeChatHistory();
        
        // Make sure the current chat is set as active
        setActiveChat(window.conversationId);
    }, 500);
};

// Initialize important global variables
window.conversationId = window.conversationId || 'chat_' + Date.now(); // Ensure we have a valid conversation ID
window.messageCount = 0;
// Add flag to prevent double submission
let isSending = false;

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded - initializing chat application");
    
    // CLEAR ALL CHAT HISTORY ON PAGE LOAD/REFRESH
    console.log("CLEARING ALL CHAT HISTORY STORAGE");
    localStorage.removeItem('abby_chat_history');
    
    // FORCE SHOW TUTORIAL ON EVERY PAGE LOAD
    console.log("FORCING TUTORIAL TO SHOW");
    localStorage.removeItem('abby_tutorial_completed');
    sessionStorage.setItem('force_show_tutorial', 'true');
    
    // Always create a new conversation ID on page load/refresh
    window.conversationId = 'chat_' + Date.now();
    console.log("New session created with conversation ID:", window.conversationId);
    
    window.messageCount = 0;
    
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const suggestedPrompts = document.getElementById('suggestedPrompts');
    const promptButtons = document.querySelectorAll('.prompt-btn');
    
    // Clear any existing chat content on page load
    if (chatMessages) {
        console.log("Clearing any existing chat content");
        chatMessages.innerHTML = '';
    }
    
    // Initialize the chat history display with a clean slate
    initializeChatHistory();
    
    // Initialize the sidebar new chat button
    console.log("Initializing sidebar new chat button");
    const newChatButton = document.getElementById('newChatButton');
    if (newChatButton) {
        console.log("New chat button found, attaching event listener");
        // Remove existing click listeners to prevent duplicates
        const newButtonClone = newChatButton.cloneNode(true);
        newChatButton.parentNode.replaceChild(newButtonClone, newChatButton);
        
        // Add fresh click listener with debugging
        newButtonClone.addEventListener('click', function(e) {
            e.preventDefault();
            console.log("New Chat button clicked, starting new chat");
            window.startNewChat();
        });
    } else {
        console.error("New chat button not found");
    }
    
    // Track message count for $1 per chat limit
    const MAX_MESSAGES_PER_CHAT = 20; // $1 represents about 20 messages
    
    // Auto-expand textarea as user types
    userInput.addEventListener('input', function() {
        // Enable/disable send button based on content
        sendButton.disabled = !userInput.value.trim();
        
        // Hide suggested prompts when user starts typing
        if (userInput.value.trim()) {
            const suggestedPrompts = document.getElementById('suggestedPrompts');
            if (suggestedPrompts) {
                suggestedPrompts.style.display = 'none';
            }
        }
        
        // Auto-expand textarea (reset height first to get accurate scrollHeight)
        this.style.height = 'auto';
        const newHeight = Math.min(this.scrollHeight, 120); // Cap at 120px
        this.style.height = newHeight + 'px';
    });

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message && !isSending) { // Check isSending flag
            console.log("Form submitted with message:", message);
            
            // Hide suggestion prompts
            const suggestedPrompts = document.getElementById('suggestedPrompts');
            if (suggestedPrompts) {
                suggestedPrompts.style.display = 'none';
            }
            
            sendMessage(message);
            userInput.value = '';
            sendButton.disabled = true;
            userInput.style.height = 'auto';
        }
    });
    
    // Handle Enter key to send message (with Shift+Enter for new line)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default to avoid form submission
            // Let the form submission handle it to prevent double processing
            // Only trigger sendMessage if not already submitting the form
            if (!e.isComposing && !e.repeat) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Handle suggestion prompt buttons
    promptButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (isSending) return; // Check isSending flag
            
            const promptText = this.getAttribute('data-prompt');
            userInput.value = promptText;
            sendButton.disabled = false;
            // Focus on input so user can modify if desired
            userInput.focus();
            
            // Auto-send after a short delay if user doesn't modify
            setTimeout(() => {
                if (userInput.value === promptText && !isSending) { // Check isSending flag
                    sendMessage(promptText);
                    userInput.value = '';
                    sendButton.disabled = true;
                }
            }, 800);
        });
    });

    // Initialize chat with a welcome message (without feedback options)
    addBotWelcomeMessage("Hi! 👋 How can I help you today?");

    // Functions for chat interaction
    function addUserMessage(message) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.style.alignSelf = 'flex-end';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Your message');

        const messageEl = document.createElement('div');
        messageEl.className = 'message user-message latest';
        messageEl.textContent = message;
        
        // Add timestamp to message
        const timestamp = document.createElement('span');
        timestamp.className = 'message-time';
        const now = new Date();
        timestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageEl.appendChild(timestamp);

        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
        
        // Remove 'latest' class after animation completes
        setTimeout(() => {
            messageEl.classList.remove('latest');
        }, 2000);
    }

    // Expose sendMessage function for use by the tutorial
    window.sendMessageToAbby = function(message) {
        if (message && message.trim() && !isSending) { // Check isSending flag
            sendMessage(message.trim());
        } else if (isSending) {
            console.log("Already sending a message, tutorial message ignored");
        }
    };

    // Function to format and add citations
    function processCitations(citation_objects, formattedMessage) {
        // Check if we have citation data
        if (!citation_objects || !Array.isArray(citation_objects) || citation_objects.length === 0) {
            return { formattedMessage, citationsHTML: '' };
        }
        
        // Create citations section
        let citationsHTML = '<div class="citations-section"><h4>Sources:</h4><ol class="citation-list">';
        
        // Map to track which citations have been added
        const addedCitations = new Set();
        const addedUrls = new Set();
        
        // Process each citation
        citation_objects.forEach((citation, index) => {
            // Skip null or undefined citations
            if (!citation) {
                return;
            }
            
            // Generate a unique citation ID
            const citationId = citation.id || `citation-${index + 1}`;
            
            // Skip if we've already added this citation
            if (addedCitations.has(citationId)) {
                return;
            }
            
            // Skip if URL is already added (avoid duplicates)
            if (citation.url && addedUrls.has(citation.url)) {
                return;
            }
            
            // Add to tracking sets
            addedCitations.add(citationId);
            if (citation.url) {
                addedUrls.add(citation.url);
            }
            
            // Get citation details
            const title = citation.title || citation.source || 'Unknown Source';
            const url = citation.url || 'https://www.abortionfinder.org/';
            const author = citation.author || '';
            const date = citation.date || citation.accessed_date || '';
            const publisher = citation.publisher || '';
            
            // Format citation details
            let citationDetails = '';
            if (author) citationDetails += author;
            if (date) citationDetails += (citationDetails ? ', ' : '') + formatDate(date);
            if (publisher) citationDetails += (citationDetails ? '. ' : '') + publisher;
            
            // Add citation to the list
            citationsHTML += `
                <li id="${citationId}" class="citation-item">
                    <a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link">
                        <span class="citation-title">${title}</span>
                    </a>
                    ${citationDetails ? `<div class="citation-details">${citationDetails}</div>` : ''}
                </li>
            `;
        });
        
        // Close the citations section
        citationsHTML += '</ol></div>';
        
        // Only return non-empty citations
        if (addedCitations.size === 0) {
            return { formattedMessage, citationsHTML: '' };
        }
        
        return { formattedMessage, citationsHTML };
    }

    // Format date string
    function formatDate(dateStr) {
        // Try to parse the date
        try {
            const date = new Date(dateStr);
            if (!isNaN(date.getTime())) {
                return date.toLocaleDateString('en-US', { 
                    year: 'numeric', 
                    month: 'long', 
                    day: 'numeric' 
                });
            }
        } catch (e) {
            // If parsing fails, return the original string
        }
        return dateStr;
    }

    // Function to add bot message to the chat
    function addBotMessage(message, citations = null, citation_objects = null, graphics = null) {
        // TEMPORARY DEBUG: Log the incoming message
        console.log("DEBUG - addBotMessage input:", {
            messageType: typeof message,
            messageValue: message,
            hasToString: message && typeof message.toString === 'function',
            citations: citations, 
            citation_objects: citation_objects
        });
        
        // Create message container
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Abby\'s response');
        
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message latest';
        
        // Remove previous 'latest' classes
        document.querySelectorAll('.message.latest').forEach(msg => {
            if (msg !== messageEl) {
                msg.classList.remove('latest');
            }
        });
        
        // Ensure message is a string
        let messageText = '';
        if (typeof message === 'string') {
            messageText = message;
        } else if (message && typeof message.toString === 'function') {
            messageText = message.toString();
        } else {
            messageText = "I'm sorry, I couldn't generate a response.";
        }
        
        // ***** FIX: Trust that the message from backend is valid HTML *****
        // Simple cleanups to remove any stray blockquote markers that might remain
        messageText = messageText
            .replace(/^>\s*•\s+/gm, '• ')
            .replace(/^>\s+/gm, '')
            .replace(/^>/gm, '');
        
        // Set content directly using innerHTML since we trust the backend HTML formatting
        messageEl.innerHTML = messageText;
        // ***** END FIX *****
        
        // Add timestamp to message
        const timestamp = document.createElement('span');
        timestamp.className = 'message-time';
        const now = new Date();
        timestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Set message content using innerHTML for HTML support
        try {
            // Process citations if available
            const { formattedMessage, citationsHTML } = processCitations(citation_objects, messageText);
            
            // Set the message content
            messageEl.innerHTML = formattedMessage;
            
            // Append timestamp
            messageEl.appendChild(timestamp);
            
            // Add citations HTML if available
            if (citationsHTML) {
                const citationsContainer = document.createElement('div');
                citationsContainer.innerHTML = citationsHTML;
                messageEl.appendChild(citationsContainer);
            }
        } catch (e) {
            console.error("Error setting innerHTML:", e);
            messageEl.textContent = "I'm sorry, I couldn't process your request. Please try again.";
            messageEl.appendChild(timestamp);
        }
        
        // Add message ID as a data attribute for feedback
        const messageId = 'msg_' + new Date().getTime();
        messageEl.dataset.messageId = messageId;
        
        // Add the feedback controls
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'feedback-container';
        feedbackContainer.dataset.messageId = messageId;
        feedbackContainer.style.display = 'flex';
        feedbackContainer.style.alignItems = 'center'; // Align items vertically center
        
        const feedbackPrompt = document.createElement('div');
        feedbackPrompt.className = 'feedback-prompt';
        feedbackPrompt.textContent = 'Was this response helpful?';
        feedbackPrompt.style.marginBottom = '0'; // Remove bottom margin
        feedbackPrompt.style.marginRight = '3px'; // Reduced from 10px to 3px to keep thumbs closer to text
        
        // Create the feedback buttons with less spacing
        const feedbackButtons = document.createElement('div');
        feedbackButtons.className = 'feedback-buttons';
        feedbackButtons.style.display = 'flex';
        feedbackButtons.style.gap = '0px'; // Reduced to 0px for no gap
        feedbackButtons.style.alignItems = 'center'; // Center buttons vertically
        
        const thumbsUpButton = document.createElement('button');
        thumbsUpButton.className = 'feedback-button thumbs-up';
        thumbsUpButton.innerHTML = '<i class="fas fa-thumbs-up"></i>';
        thumbsUpButton.setAttribute('aria-label', 'Yes, this was helpful');
        thumbsUpButton.style.border = 'none';
        thumbsUpButton.style.outline = 'none';
        thumbsUpButton.style.background = 'transparent';
        thumbsUpButton.style.boxShadow = 'none';
        thumbsUpButton.style.padding = '10px'; // Add padding to increase the clickable area
        
        const thumbsDownButton = document.createElement('button');
        thumbsDownButton.className = 'feedback-button thumbs-down';
        thumbsDownButton.innerHTML = '<i class="fas fa-thumbs-down"></i>';
        thumbsDownButton.setAttribute('aria-label', 'No, this was not helpful');
        thumbsDownButton.style.border = 'none';
        thumbsDownButton.style.outline = 'none';
        thumbsDownButton.style.background = 'transparent';
        thumbsDownButton.style.boxShadow = 'none';
        thumbsDownButton.style.padding = '10px'; // Add padding to increase the clickable area
        
        // Add event listeners for feedback buttons
        thumbsUpButton.addEventListener('click', function() {
            // Record positive feedback
            submitFeedback(messageId, 5);
            
            // Disable the buttons
            thumbsUpButton.disabled = true;
            thumbsDownButton.disabled = true;
            
            // Update styles to show selection
            thumbsUpButton.classList.add('selected');
            
            // Show "Thank you" message
            feedbackPrompt.textContent = 'Thank you for your feedback!';
            feedbackButtons.style.display = 'none';
        });
        
        thumbsDownButton.addEventListener('click', function() {
            // Show detailed feedback options when user selects thumbs down
            showDetailedFeedback(feedbackContainer, false);
        });
        
        // Assemble the feedback container
        feedbackButtons.appendChild(thumbsUpButton);
        feedbackButtons.appendChild(thumbsDownButton);
        
        feedbackContainer.appendChild(feedbackPrompt);
        feedbackContainer.appendChild(feedbackButtons);
        
        // Add everything to the DOM
        messageContainer.appendChild(messageEl);
        messageContainer.appendChild(feedbackContainer);
        
        // Get the chat messages container and append
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.appendChild(messageContainer);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        return messageEl;
    }

    function addTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.id = 'typingIndicator';
        
        // Create text container with initial text
        const textContainer = document.createElement('div');
        textContainer.className = 'thinking-text';
        textContainer.textContent = 'Generating response.';
        typingIndicator.appendChild(textContainer);
        
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
        
        // Start the dot animation
        let dotCount = 1; // Start with one dot
        window.thinkingAnimation = setInterval(() => {
            dotCount = (dotCount % 3) + 1; // Cycle between 1, 2, and 3 dots
            let dots = '.'.repeat(dotCount);
            textContainer.textContent = 'Generating response' + dots;
        }, 500); // Update every 500ms
    }

    function removeTypingIndicator() {
        // Clear the interval to stop the animation
        if (window.thinkingAnimation) {
            clearInterval(window.thinkingAnimation);
            window.thinkingAnimation = null;
        }
        
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function addBotWelcomeMessage(message) {
        // Clear any existing messages first to ensure clean slate
        chatMessages.innerHTML = '';
        
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container welcome-message-container';
        messageContainer.setAttribute('role', 'listitem');
        messageContainer.setAttribute('aria-label', 'Welcome message');
        messageContainer.style.width = '100%';
        messageContainer.style.marginTop = '20px'; // Add space at top

        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message welcome-message';
        messageEl.style.maxWidth = 'none';
        messageEl.style.width = 'auto';
        
        // Convert markdown-like syntax to HTML
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>') // Code
            .replace(/\n\n/g, '<br><br>') // Line breaks
            .replace(/\n/g, '<br>'); // Line breaks
        
        messageEl.innerHTML = formattedMessage;
        
        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        
        // Force scroll to top to ensure welcome message is visible
        chatMessages.scrollTop = 0;
        
        // Add suggestion prompts
        addSuggestionPrompts();
        
        // Ensure everything is visible
        setTimeout(() => {
            chatMessages.scrollTop = 0;
        }, 100);
    }

    function addSuggestionPrompts() {
        const suggestedPromptsContainer = document.createElement('div');
        suggestedPromptsContainer.className = 'suggested-prompts';
        suggestedPromptsContainer.id = 'suggestedPrompts';
        
        const grid = document.createElement('div');
        grid.className = 'prompts-grid';
        
        const suggestions = [
            { text: 'Can I get an abortion in my state?', emoji: '🗺️' },
            { text: 'What contraception methods are available?', emoji: '💊' },
            { text: 'How does pregnancy happen?', emoji: '🤰' },
            { text: 'What are some stress management tips?', emoji: '🧘' },
            { text: 'Explain STI prevention', emoji: '🛡️' },
            { text: 'What are the signs of pregnancy?', emoji: '🔍' }
        ];
        
        // Create one button per row instead of two buttons per row
        suggestions.forEach(suggestion => {
            // Create button
            const btn = document.createElement('button');
            btn.className = 'prompt-btn';
            btn.setAttribute('data-prompt', suggestion.text);
            
            // Create text and emoji span for proper positioning
            const textSpan = document.createElement('span');
            textSpan.className = 'prompt-text';
            textSpan.textContent = suggestion.text;
            
            const emojiSpan = document.createElement('span');
            emojiSpan.className = 'prompt-emoji';
            emojiSpan.textContent = suggestion.emoji;
            
            // Add emoji first, then text (emoji on left)
            btn.appendChild(emojiSpan);
            btn.appendChild(textSpan);
            
            // Add button directly to grid (no row container)
            grid.appendChild(btn);
        });
        
        suggestedPromptsContainer.appendChild(grid);
        chatMessages.appendChild(suggestedPromptsContainer);
        
        // Add event listeners to the suggestion buttons
        const promptButtons = suggestedPromptsContainer.querySelectorAll('.prompt-btn');
        promptButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Add isSending check here too
                if (isSending) return;
                
                const promptText = this.getAttribute('data-prompt');
                sendMessage(promptText);
                
                // Hide suggestions after clicking
                suggestedPromptsContainer.style.display = 'none';
            });
        });
    }
    
    function scrollToBottom() {
        // Don't scroll if we only have the welcome message
        const messageContainers = chatMessages.querySelectorAll('.message-container:not(.welcome-message-container)');
        if (messageContainers.length === 0) {
            return;
        }
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to detect location-based queries and extract ZIP codes
    function detectLocationBasedQuery(message) {
        // Check if the message is about finding abortion clinics
        const clinicKeywords = ['abortion clinic', 'abortion provider', 'planned parenthood', 'family planning', 'women\'s health', 'abortion care'];
        
        // Check if the message is about abortion policy
        const policyKeywords = [
            'abortion policy', 'abortion law', 'abortion legal', 'legal abortion', 
            'policy on abortion', 'abortion restriction', 'can I get an abortion',
            'is abortion legal', 'abortion banned', 'abortion allowed', 'abortion access',
            'get an abortion in', 'abortion laws in', 'abortion rights', 'right to abortion',
            'terminate pregnancy', 'abortion rules', 'abortion regulations'
        ];
        
        const locationKeywords = ['near', 'in', 'around', 'close to', 'nearby', 'at', 'within', 'for'];
        
        // Convert message to lowercase for case-insensitive matching
        const lowercaseMessage = message.toLowerCase();
        
        // Check if message contains clinic keywords
        const hasClinicKeyword = clinicKeywords.some(keyword => lowercaseMessage.includes(keyword));
        
        // Check if message contains policy keywords
        const hasPolicyKeyword = policyKeywords.some(keyword => lowercaseMessage.includes(keyword));
        
        // Check if the message directly asks about abortion legality
        const legalityPattern = /\b(is|are|can)\s+(abortion|abortions)\s+(legal|allowed|banned|restricted|available|accessible)\b/i;
        const hasLegalityQuestion = legalityPattern.test(message);
        
        // Check if message contains location keywords
        const hasLocationKeyword = locationKeywords.some(keyword => lowercaseMessage.includes(keyword));
        
        // Check for state name directly
        const statePattern = /\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b/i;
        const stateMatch = message.match(statePattern);
        const hasStateName = !!stateMatch;
        
        // Check for ZIP code using regex (5-digit US ZIP code)
        const zipCodeMatch = message.match(/\b\d{5}\b/);
        const zipCode = zipCodeMatch ? zipCodeMatch[0] : null;
        
        // Check for city name (with improved regex for complex queries)
        let cityMatch = null;
        for (const keyword of locationKeywords) {
            // This pattern looks for a location keyword followed by one or more words,
            // and doesn't require an immediate end of sentence or comma
            const cityRegex = new RegExp(`${keyword}\\s+([A-Za-z][A-Za-z\\s]+?)(?:\\s+(?:that|which|and|or|but|the|if|is|are|can|will|to)|[,.]|$)`, 'i');
            const match = message.match(cityRegex);
            if (match && match[1]) {
                cityMatch = match[1].trim();
                break;
            }
        }
        
        // Enhanced detection for "abortion in [location]" type queries
        const abortionInLocationPattern = /\b(abortion|abortions)\s+(in|at|for)\s+([A-Za-z][A-Za-z\s]+|\d{5})\b/i;
        const abortionInLocationMatch = message.match(abortionInLocationPattern);
        const hasAbortionInLocation = !!abortionInLocationMatch;
        
        // Special case for "can I get an abortion in [location]" which is both policy and location
        const canGetAbortionPattern = /\bcan\s+(?:i|you|someone|a\s+woman|a\s+person)\s+get\s+(?:an\s+)?abortion\s+(?:in|at|near)\s+/i;
        const canGetAbortionMatch = canGetAbortionPattern.test(message);
        
        // Return result
        return {
            isLocationClinicQuery: hasClinicKeyword && (hasLocationKeyword || zipCode || hasStateName),
            isLocationPolicyQuery: (hasPolicyKeyword || hasLegalityQuestion || canGetAbortionMatch || 
                                   (lowercaseMessage.includes('abortion') && hasAbortionInLocation)) && 
                                  (hasLocationKeyword || zipCode || hasStateName),
            zipCode: zipCode,
            cityName: cityMatch,
            stateName: stateMatch ? stateMatch[0] : null
        };
    }

    // For backward compatibility, create an alias for the old function name
    function detectLocationBasedClinicQuery(message) {
        return detectLocationBasedQuery(message);
    }

    function sendMessage(message) {
        if (!message.trim()) {
            return;
        }
        
        // Check if already sending a message to prevent double submissions
        if (isSending) {
            console.log("Already sending a message, please wait...");
            return;
        }
        
        // Set flag to indicate we're processing a message - SET THIS IMMEDIATELY
        isSending = true;
        
        try {
            console.log("=== DEBUGGING CHAT ===");
            console.log("1. Starting sendMessage with message:", message);
            
            // Add user message to chat
            addUserMessage(message);
            
            // Add typing indicator
            addTypingIndicator();
            
            // Disable input while processing
            document.getElementById('userInput').setAttribute('disabled', 'disabled');
            document.getElementById('sendButton').setAttribute('disabled', 'disabled');
            
            // Get the current session ID
            const sessionId = getCurrentSessionId();
            
            // Check if this is a location-based query
            const queryInfo = detectLocationBasedQuery(message);
            
            // Log the request information
            console.log(`2. Sending message to server. Session ID: ${sessionId}, Message: ${message}`);
            
            // Prepare data for the request
            const data = {
                message: message,
                session_id: sessionId
            };
            
            // Include location information if detected
            if (queryInfo.isLocationClinicQuery || queryInfo.isLocationPolicyQuery) {
                if (queryInfo.isLocationClinicQuery) {
                    data.is_location_clinic_query = true;
                }
                if (queryInfo.isLocationPolicyQuery) {
                    data.is_location_policy_query = true;
                }
                if (queryInfo.zipCode) {
                    data.zip_code = queryInfo.zipCode;
                }
                if (queryInfo.cityName) {
                    data.city_name = queryInfo.cityName;
                }
                if (queryInfo.stateName) {
                    data.state_name = queryInfo.stateName;
                }
            }
            
            // Optional: Add user location if available
            if (window.userLocation) {
                data.user_location = window.userLocation;
                console.log('3. Including user location in request:', window.userLocation);
            }
            
            console.log("4. Request payload:", JSON.stringify(data));
            
            // Send request to the server using fetch instead of jQuery
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(response => {
                try {
                    // TEMPORARY DEBUG: Log the raw response from fetch
                    console.log("DEBUG - AJAX raw response:", response);
                    console.log("DEBUG - AJAX response type:", typeof response);
                    
                    // Remove typing indicator
                    removeTypingIndicator();
                    
                    // Re-enable input
                    document.getElementById('userInput').removeAttribute('disabled');
                    document.getElementById('sendButton').removeAttribute('disabled');
                    
                    // Focus the input field for the next message
                    document.getElementById('userInput').focus();
                    
                    // Check if response is null or undefined
                    if (!response) {
                        console.error("AJAX Error: Received null response from server");
                        addBotMessage("I'm sorry, I couldn't process your request right now. Please try again later.");
                        return;
                    }
                    
                    // Get response text
                    let messageText = null;
                    let citations = [];
                    let citation_objects = [];
                    
                    // Handle string response
                    if (typeof response === 'string') {
                        console.log("DEBUG - Response is a string");
                        messageText = response;
                    }
                    // Handle object response
                    else if (typeof response === 'object') {
                        messageText = response.text || response.message || "I'm sorry, I couldn't generate a response.";
                        
                        // Extract citations if available
                        if (Array.isArray(response.citations)) {
                            citations = response.citations;
                        }
                        if (Array.isArray(response.citation_objects)) {
                            citation_objects = response.citation_objects;
                        } else if (typeof response.citation_objects === 'string') {
                            // Try to parse if it's a string
                            try {
                                citation_objects = JSON.parse(response.citation_objects);
                            } catch (e) {
                                console.error("Failed to parse citation_objects string:", e);
                            }
                        }
                    }
                    
                    console.log("DEBUG - Extracted message text:", messageText);
                    
                    // Add the bot message to chat
                    const messageEl = addBotMessage(messageText, citations, citation_objects);
                    
                    // Handle map display - Check both server response and local detection
                    const isPPCitationOnly = citation_objects && 
                                         citation_objects.length > 0 && 
                                         citation_objects.some(c => c.source === "Planned Parenthood") &&
                                         !messageText.toLowerCase().includes("clinic") &&
                                         !messageText.toLowerCase().includes("provider");
                    
                    // Check if message contains Planned Parenthood URLs (indicating it's likely just citations)
                    const hasPPURL = typeof messageText === 'string' && 
                                   (messageText.includes('plannedparenthood.org') || 
                                    messageText.includes('Planned Parenthood:'));
                    
                    // Check if this is a policy response where we might want to show a map
                    const isPolicyResponse = typeof messageText === 'string' && 
                                          (messageText.toLowerCase().includes('policy') ||
                                           messageText.toLowerCase().includes('laws') ||
                                           messageText.toLowerCase().includes('legal') ||
                                           messageText.toLowerCase().includes('banned')) &&
                                          messageText.toLowerCase().includes('abortion');
                    
                    // Check if this is a restrictive state response that mentions travel
                    const isRestrictiveState = isPolicyResponse && 
                                             (messageText.toLowerCase().includes('not legal') || 
                                              messageText.toLowerCase().includes('banned') ||
                                              messageText.toLowerCase().includes('prohibited') ||
                                              messageText.toLowerCase().includes('restricted') ||
                                              messageText.toLowerCase().includes('seeking abortion care') ||
                                              messageText.toLowerCase().includes('travel'));
                                          
                    // Show map for policy responses if server specifically requests it or we have location info
                    const showMapForPolicy = isPolicyResponse && 
                                          (response.show_map === true || 
                                           response.travel_state ||
                                           (queryInfo.isLocationPolicyQuery && 
                                            (queryInfo.zipCode || queryInfo.cityName || queryInfo.stateName || response.zip_code)));
                                     
                    // Only show map if explicitly requested or for clinic queries, not for information-only queries
                    const shouldShowMap = (response.show_map === true) || 
                                         showMapForPolicy ||
                                         (queryInfo.isLocationClinicQuery && 
                                          (queryInfo.zipCode || queryInfo.cityName || queryInfo.stateName) &&
                                          !isPPCitationOnly && 
                                          !hasPPURL);
                                         
                    const locationToShow = response.zip_code || queryInfo.zipCode || queryInfo.cityName || queryInfo.stateName;
                    const mapQuery = response.map_query || (isPolicyResponse ? 
                                                         `abortion clinics near ${locationToShow}` : 
                                                         `abortion clinic in ${locationToShow}`);
                    
                    console.log("DEBUG - Map display check:", {
                        shouldShowMap,
                        isPPCitationOnly,
                        hasPPURL,
                        isPolicyResponse,
                        isRestrictiveState,
                        hasTravel: !!response.travel_state,
                        showMapForPolicy,
                        responseShowMap: response.show_map,
                        isLocationQuery: queryInfo.isLocationClinicQuery,
                        isLocationPolicyQuery: queryInfo.isLocationPolicyQuery,
                        locationToShow
                    });
                    
                    if (shouldShowMap && locationToShow && messageEl) {
                        console.log("DEBUG - Showing map for location:", locationToShow);
                        
                        try {
                            // Create a unique ID for the map
                            const mapId = 'clinic-map-' + Date.now();
                            const mapContainer = document.createElement('div');
                            mapContainer.style.width = '100%'; 
                            mapContainer.style.height = '300px';
                            mapContainer.style.marginTop = '10px';
                            mapContainer.style.marginBottom = '10px';
                            mapContainer.className = 'map-wrapper';
                            mapContainer.style.display = 'block';
                            mapContainer.innerHTML = `<div id="${mapId}" class="clinic-map" style="width: 100%; height: 100%;"></div>`;
                            
                            // Add map container to message element
                            messageEl.appendChild(mapContainer);
                            
                            // Special handling for travel routes
                            let mapSrc = "";
                            if (response.travel_state) {
                                // Generate directions from the current state to the travel state
                                const origin = response.state_name || locationToShow;
                                const destination = response.travel_state;
                                
                                // Use directions mode for Google Maps
                                mapSrc = `https://www.google.com/maps/embed/v1/directions?key=${window.GOOGLE_MAPS_API_KEY}`+
                                         `&origin=${encodeURIComponent(origin)}`+
                                         `&destination=${encodeURIComponent(destination)}`+
                                         `&mode=driving`;
                            } else {
                                // Standard place search
                                mapSrc = `https://www.google.com/maps/embed/v1/search?key=${window.GOOGLE_MAPS_API_KEY}&q=${encodeURIComponent(mapQuery)}`;
                            }
                            
                            // Use Google Maps Embed API for a simple solution
                            const googleMapsEmbed = document.createElement('iframe');
                            googleMapsEmbed.width = '100%';
                            googleMapsEmbed.height = '100%';
                            googleMapsEmbed.style.border = '0';
                            googleMapsEmbed.style.borderRadius = '12px';
                            googleMapsEmbed.setAttribute('loading', 'lazy');
                            googleMapsEmbed.setAttribute('allowfullscreen', '');
                            googleMapsEmbed.setAttribute('referrerpolicy', 'no-referrer-when-downgrade');
                            googleMapsEmbed.src = mapSrc;
                            
                            // Replace the placeholder with the iframe
                            const mapElement = document.getElementById(mapId);
                            if (mapElement) {
                                mapElement.innerHTML = '';
                                mapElement.appendChild(googleMapsEmbed);
                            }
                            
                            // Fallback if embed doesn't work - use the more complex maps API
                            googleMapsEmbed.onerror = function() {
                                console.error("Google Maps Embed API failed, trying Maps JavaScript API");
                                if (window.mapsApi && typeof window.mapsApi.showClinicMap === 'function') {
                                    try {
                                        window.mapsApi.showClinicMap(locationToShow, mapId);
                                    } catch (err) {
                                        console.error("Error showing map with Maps API:", err);
                                        mapContainer.innerHTML = getFallbackMapContent(locationToShow, response.is_legal);
                                    }
                                } else {
                                    mapContainer.innerHTML = getFallbackMapContent(locationToShow, response.is_legal);
                                }
                            };
                        } catch (mapError) {
                            console.error("Error displaying map:", mapError);
                        }
                    } else if (queryInfo.isLocationClinicQuery && !locationToShow) {
                        // Handle case where location query is detected but no ZIP/city found
                        console.log("DEBUG - Location clinic query detected but no location specified");
                        
                        // Add a follow-up message asking for ZIP code
                        setTimeout(() => {
                            addBotMessage("Please share your ZIP code or city name so I can find nearby abortion clinics.");
                        }, 1000);
                    }
                    
                    // Save the chat history for this session
                    saveChatToHistory(sessionId);
                    
                    // Update chat history in sidebar
                    updateChatHistory();
                    
                    // Track message
                    if (typeof trackEvent === 'function') {
                        trackEvent('chat_message_received');
                    }
                } catch (innerError) {
                    console.error("Error in success handler:", innerError);
                    console.error("Error stack:", innerError.stack);
                    addBotMessage("I'm sorry, something went wrong while displaying my response. Please try again.");
                }
            })
            .catch(error => {
                console.error("Fetch error:", error);
                console.error("Error stack:", error.stack);
                
                // Remove typing indicator
                removeTypingIndicator();
                
                // Re-enable input
                document.getElementById('userInput').removeAttribute('disabled');
                document.getElementById('sendButton').removeAttribute('disabled');
                
                // Focus the input field
                document.getElementById('userInput').focus();
                
                // Show error message
                addBotMessage("I'm sorry, but I couldn't connect to the server. Please check your internet connection and try again.");
            })
            .finally(() => {
                // Reset the sending flag to allow future messages
                isSending = false;
            });
        } catch (error) {
            console.error("Error in sendMessage:", error);
            console.error("Error stack:", error.stack);
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Re-enable input
            document.getElementById('userInput').removeAttribute('disabled');
            document.getElementById('sendButton').removeAttribute('disabled');
            
            // Focus the input field
            document.getElementById('userInput').focus();
            
            // Show error message
            addBotMessage("I'm sorry, something went wrong. Please try again.");
            
            // Reset the sending flag in case of error
            isSending = false;
        }
    }

    // Fallback function for map content when API fails
    function getFallbackMapContent(location, isLegal = true) {
        // Different content based on whether abortion is legal in the location
        if (isLegal === false) {
            return `
                <div style="padding: 20px; text-align: center; background-color: #f8f9fa; border-radius: 8px;">
                    <h4>Abortion Resources for ${location}</h4>
                    <p>Abortion may not be legal or may be severely restricted in ${location}.</p>
                    <p>Consider contacting these national resources for guidance:</p>
                    
                    <div style="margin-top: 15px; text-align: left; padding: 10px; background: #fff; border-radius: 6px; border-left: 4px solid #6c757d;">
                        <strong>National Abortion Federation Hotline</strong><br>
                        1-800-772-9100<br>
                        <a href="https://prochoice.org" target="_blank">prochoice.org</a>
                    </div>
                    
                    <div style="margin-top: 10px; text-align: left; padding: 10px; background: #fff; border-radius: 6px; border-left: 4px solid #6c757d;">
                        <strong>Planned Parenthood</strong><br>
                        1-800-230-PLAN<br>
                        <a href="https://www.plannedparenthood.org" target="_blank">plannedparenthood.org</a>
                    </div>
                    
                    <p style="margin-top: 15px; font-style: italic;">These organizations can help you find care in states where abortion is legal.</p>
                </div>
            `;
        } else {
            return `
                <div style="padding: 20px; text-align: center; background-color: #f8f9fa; border-radius: 8px;">
                    <h4>Abortion Clinics Near ${location}</h4>
                    <p><strong>Planned Parenthood</strong><br>
                    1691 The Alameda, San Jose, CA 95126<br>
                    Phone: (408) 287-7532</p>
                    
                    <p><strong>Women's Options Center</strong><br>
                    751 S Bascom Ave, San Jose, CA 95128<br>
                    Phone: (408) 885-2400</p>
                    
                    <p>For more options, visit: <a href="https://www.plannedparenthood.org/health-center" target="_blank">plannedparenthood.org</a></p>
                </div>
            `;
        }
    }

    function submitFeedback(messageId, rating, comment = null) {
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating,
                comment: comment
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Feedback submitted successfully:', data);
            // Hide the feedback form after submission
            const feedbackContainer = document.querySelector(`.feedback-container[data-message-id="${messageId}"]`);
            if (feedbackContainer) {
                feedbackContainer.innerHTML = '<div class="feedback-thanks">Thank you for your feedback!</div>';
            }
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
            alert('There was a problem submitting your feedback. Please try again.');
        });
    }

    // Focus input field on page load
    userInput.focus();

    function showChatLimitMessage() {
        // Create a limit reached message
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container system-message-container';
        
        const messageEl = document.createElement('div');
        messageEl.className = 'message system-message limit-message';
        messageEl.innerHTML = `
            <div class="limit-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <h3>Message limit reached</h3>
            </div>
            <p>You've reached the $1 message limit for this chat. To continue the conversation, please start a new chat.</p>
            <div class="limit-actions">
                <button id="limitNewChatBtn" class="new-chat-limit-btn">Start New Chat</button>
            </div>
        `;
        
        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
        
        // Disable the text input
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Replace textarea with a message about starting a new chat
        const chatForm = document.getElementById('chatForm');
        const inputContainer = document.createElement('div');
        inputContainer.className = 'limit-input-message';
        inputContainer.innerHTML = `
            <span>Chat limit reached. Start a new chat to continue.</span>
            <button id="footerNewChatBtn" class="footer-new-chat-btn">New Chat</button>
        `;
        
        // Replace the form with this message
        chatForm.innerHTML = '';
        chatForm.appendChild(inputContainer);
        
        // Add event listeners to both new chat buttons
        document.getElementById('limitNewChatBtn').addEventListener('click', startNewChat);
        document.getElementById('footerNewChatBtn').addEventListener('click', startNewChat);
    }

    // Clear session history
    function clearSessionHistory(skipReload = false) {
        if (window.conversationId) {
            fetch(`/session`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    session_id: window.conversationId 
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to clear conversation history');
                }
                return response.json();
            })
            .then(data => {
                console.log('Session cleared successfully:', data);
                window.conversationId = null;
                
                // Reload the page to start fresh (unless skipReload is true)
                if (!skipReload) {
                    setTimeout(() => {
                        location.reload();
                    }, 1000);
                }
            })
            .catch(error => {
                console.error('Error clearing session:', error);
            });
        } else if (!skipReload) {
            // If no conversation ID, just reload
            location.reload();
        }
    }

    // Reset tutorial and show it
    function resetTutorial() {
        console.log("Resetting tutorial...");
        // Directly clear localStorage
        localStorage.removeItem('abby_tutorial_completed');
        
        // Set a flag to indicate we want to show the tutorial
        sessionStorage.setItem('force_show_tutorial', 'true');
        
        // Reload the page
        location.reload();
    }

    // Function to show detailed feedback options after clicking thumbs down
    function showDetailedFeedback(container, isPositive) {
        // Clear existing content in the feedback container
        container.innerHTML = '';
        
        if (isPositive) {
            // Simple thank you message for positive feedback
            const thankYouMessage = document.createElement('div');
            thankYouMessage.className = 'feedback-thank-you';
            thankYouMessage.textContent = 'Thank you for your feedback!';
            container.appendChild(thankYouMessage);
        } else {
            // Create detailed feedback UI for negative feedback
            const feedbackMessage = document.createElement('div');
            feedbackMessage.className = 'feedback-message';
            feedbackMessage.textContent = 'We appreciate your feedback. How could this response be improved?';
            
            const feedbackTextarea = document.createElement('textarea');
            feedbackTextarea.className = 'feedback-textarea';
            feedbackTextarea.placeholder = 'Please share details about how we could improve this response...';
            feedbackTextarea.rows = 3;
            
            const feedbackSubmitBtn = document.createElement('button');
            feedbackSubmitBtn.className = 'feedback-submit-btn';
            feedbackSubmitBtn.textContent = 'Submit Feedback';
            feedbackSubmitBtn.addEventListener('click', function() {
                // Get the message ID from the container's data attribute
                const messageId = container.dataset.messageId;
                
                // Submit the detailed feedback
                submitFeedback(messageId, 1, feedbackTextarea.value);
                
                // Replace the feedback form with a thank you message
                container.innerHTML = '';
                const thankYouMessage = document.createElement('div');
                thankYouMessage.className = 'feedback-thank-you';
                thankYouMessage.textContent = 'Thank you for your detailed feedback!';
                container.appendChild(thankYouMessage);
            });
            
            const skipButton = document.createElement('button');
            skipButton.className = 'feedback-skip-btn';
            skipButton.textContent = 'Skip';
            skipButton.addEventListener('click', function() {
                // Replace the feedback form with a simpler thank you message
                container.innerHTML = '';
                const thankYouMessage = document.createElement('div');
                thankYouMessage.className = 'feedback-thank-you';
                thankYouMessage.textContent = 'Thank you for your feedback!';
                container.appendChild(thankYouMessage);
            });
            
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'feedback-button-container';
            buttonContainer.appendChild(feedbackSubmitBtn);
            buttonContainer.appendChild(skipButton);
            
            container.appendChild(feedbackMessage);
            container.appendChild(feedbackTextarea);
            container.appendChild(buttonContainer);
            
            // Focus the textarea for immediate input
            feedbackTextarea.focus();
        }
    }

    // Function to handle window resizing for responsive design
    function handleResponsiveLayout() {
        const width = window.innerWidth;
        const chatContainer = document.querySelector('.chat-container');
        const chatMessages = document.querySelector('.chat-messages');
        
        // Apply max-width in larger viewports
        if (width > 800) {
            chatContainer.style.maxWidth = '800px';
            chatContainer.style.margin = '0 auto';
        } else {
            chatContainer.style.maxWidth = '100%';
            chatContainer.style.margin = '0';
        }
        
        // Ensure input is visible on mobile
        if (width <= 480) {
            document.querySelector('#userInput').style.fontSize = '16px'; // Prevent zoom on mobile
        }
    }

    // Initialize responsive layout
    window.addEventListener('resize', handleResponsiveLayout);
    document.addEventListener('DOMContentLoaded', handleResponsiveLayout);

    // Initialize the chat history
    initializeChatHistory();
});

// Add this CSS to the style section
document.addEventListener('DOMContentLoaded', function() {
    // Create a style element for the new chat button
    const style = document.createElement('style');
    style.textContent = `
        .new-chat-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            background-color: var(--apple-blue);
            color: white;
            border: none;
            border-radius: 1.5rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
            z-index: 100;
        }
        
        .new-chat-btn:hover {
            background-color: #005bbf;
            transform: translateY(-1px);
        }
        
        .system-message-container {
            width: 100%;
            display: flex;
            justify-content: center;
            margin: 1rem 0;
        }
        
        .system-message {
            background-color: #f3f4f6;
            color: #374151;
            border-radius: 0.75rem;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            text-align: center;
            max-width: 80%;
            border: 1px solid #e5e7eb;
        }
        
        .limit-message {
            background-color: #fff8ee;
            border: 1px solid #ffeacc;
            padding: 1rem;
            max-width: 90%;
        }
        
        .limit-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0.5rem;
        }
        
        .limit-header svg {
            color: #f59e0b;
        }
        
        .limit-header h3 {
            font-size: 1rem;
            margin: 0;
            font-weight: 600;
            color: #78350f;
        }
        
        .limit-message p {
            margin: 0.5rem 0 1rem 0;
            color: #78350f;
        }
        
        .limit-actions {
            display: flex;
            justify-content: flex-start;
        }
        
        .new-chat-limit-btn {
            margin-top: 0.5rem;
            background-color: var(--apple-blue);
            color: white;
            border: none;
            border-radius: 1.5rem;
            padding: 0.5rem 1.2rem;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .new-chat-limit-btn:hover {
            background-color: #005bbf;
            transform: translateY(-1px);
        }
        
        .limit-input-message {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1rem;
            background-color: #fff8ee;
            border-top: 1px solid #ffeacc;
            color: #78350f;
            width: 100%;
        }
        
        .footer-new-chat-btn {
            background-color: var(--apple-blue);
            color: white;
            border: none;
            border-radius: 1.5rem;
            padding: 0.4rem 1rem;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .footer-new-chat-btn:hover {
            background-color: #005bbf;
            transform: translateY(-1px);
        }
        
        [data-bs-theme="dark"] .limit-message {
            background-color: #3a2a12;
            border-color: #634a28;
        }
        
        [data-bs-theme="dark"] .limit-header h3 {
            color: #fbd38d;
        }
        
        [data-bs-theme="dark"] .limit-message p {
            color: #f8f9fa;
        }
        
        [data-bs-theme="dark"] .limit-input-message {
            background-color: #3a2a12;
            border-color: #634a28;
            color: #f8f9fa;
        }
        
        [data-bs-theme="dark"] .system-message {
            background-color: #2c2c2e;
            color: #f5f5f7;
            border-color: #424245;
        }
        
        .show-tutorial-btn {
            position: absolute;
            top: 1rem;
            right: 7.5rem;
            padding: 0.5rem 1rem;
            background-color: #34c759;
            color: white;
            border: none;
            border-radius: 1.5rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
            z-index: 100;
        }
        
        .show-tutorial-btn:hover {
            background-color: #28a745;
            transform: translateY(-1px);
        }
    `;
    document.head.appendChild(style);
});

// Chat history management functions
function initializeChatHistory() {
    console.log("Initializing chat history");
    const historyContainer = document.getElementById('chatHistoryContainer');
    if (!historyContainer) {
        console.error('Chat history container not found');
        return;
    }
    
    // Clear any existing items
    historyContainer.innerHTML = '';
    
    // Load saved chat history from localStorage
    const savedChats = getSavedChats();
    console.log("Found saved chats:", savedChats.length);
    
    // Ensure the sidebar has proper styling
    addSidebarStyles();
    
    // Add the current conversation first
    const currentChatItem = document.createElement('div');
    currentChatItem.className = 'history-item active';
    currentChatItem.id = 'current-chat';
    currentChatItem.textContent = 'Current Conversation';
    currentChatItem.dataset.id = window.conversationId || 'current'; 
    historyContainer.appendChild(currentChatItem);
    
    // Add saved chats to the history
    if (savedChats && savedChats.length > 0) {
        console.log("Adding saved chats to history sidebar");
        savedChats.forEach((chat, index) => {
            if (chat.id !== (window.conversationId || 'current')) {
                console.log(`Adding chat #${index+1} to sidebar:`, chat.id, chat.title);
                const chatItem = document.createElement('div');
                chatItem.className = 'history-item';
                chatItem.textContent = chat.title || 'Chat ' + new Date(parseInt(chat.id.split('_')[1] || Date.now())).toLocaleDateString();
                chatItem.dataset.id = chat.id;
                chatItem.addEventListener('click', function() {
                    console.log("Loading chat from history:", chat.id);
                    loadChatFromHistory(chat.id);
                });
                historyContainer.appendChild(chatItem);
            } else {
                console.log(`Skipping chat #${index+1} (same as current):`, chat.id);
            }
        });
    } else {
        console.log("No saved chats found to display in history");
    }
    
    // Add click event to the current chat
    currentChatItem.addEventListener('click', function() {
        // Just set it active, no loading needed
        setActiveChat(this.dataset.id);
    });
}

function updateChatHistory() {
    // Update the current chat item's ID
    const currentChatItem = document.getElementById('current-chat');
    if (currentChatItem) {
        currentChatItem.dataset.id = window.conversationId || 'current';
    }
    
    // Set the current chat as active
    setActiveChat(window.conversationId || 'current');
}

function setActiveChat(chatId) {
    // Remove active class from all items
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add active class to the selected item
    document.querySelectorAll(`.history-item[data-id="${chatId}"]`).forEach(item => {
        item.classList.add('active');
    });
}

function saveChatToHistory(chatId) {
    if (!chatId) {
        console.error("Cannot save chat: No chat ID provided");
        return;
    }
    
    console.log("FORCE SAVING CHAT:", chatId);
    
    // Get the chat messages
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) {
        console.error("Cannot save chat: Chat messages element not found");
        return;
    }
    
    // REMOVED CHECK - Force save even with limited messages
    // Set a basic title
    let chatTitle = 'Chat ' + new Date().toLocaleDateString();
    
    // Get the first user message to use as title (if available)
    const userMessages = chatMessages.querySelectorAll('.user-message');
    if (userMessages.length > 0) {
        const firstUserMessageText = userMessages[0].textContent.trim();
        if (firstUserMessageText) {
            // Use the first 25 chars of user message as title, removing timestamp
            const messageWithoutTime = firstUserMessageText.replace(/\d+:\d+\s*(AM|PM)?$/i, '').trim();
            chatTitle = messageWithoutTime.substring(0, 25) + (messageWithoutTime.length > 25 ? '...' : '');
        }
    }
    
    console.log("Saving chat with title:", chatTitle);
    
    // Create chat history object
    const chatHistory = {
        id: chatId,
        title: chatTitle,
        messages: chatMessages.innerHTML || "<p>Empty chat</p>", // Force some content
        timestamp: Date.now()
    };
    
    // Save to localStorage
    saveChat(chatHistory);
    
    // FORCE refresh the chat history display immediately after saving
    console.log("FORCE REFRESH after save");
    initializeChatHistory();
}

function loadChatFromHistory(chatId) {
    console.log("Attempting to load chat:", chatId);
    
    // Save current chat first (if it has content)
    const currentChat = document.getElementById('chatMessages');
    if (window.conversationId && currentChat && currentChat.childElementCount > 0) {
        console.log("Saving current chat before loading:", window.conversationId);
        saveChatToHistory(window.conversationId);
    }
    
    // Load the selected chat
    const savedChats = getSavedChats();
    const chatToLoad = savedChats.find(chat => chat.id === chatId);
    
    if (chatToLoad) {
        console.log("Found chat to load:", chatToLoad.title);
        
        // Set as current conversation
        window.conversationId = chatToLoad.id;
        
        // Load messages
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            // Clear existing messages
            chatMessages.innerHTML = '';
            
            // Insert saved messages
            setTimeout(() => {
                chatMessages.innerHTML = chatToLoad.messages;
                console.log("Loaded chat messages");
                
                // Re-attach any event listeners to citation references
                const citationRefs = chatMessages.querySelectorAll('.citation-ref');
                if (citationRefs.length > 0) {
                    console.log("Re-attaching citation listeners");
                    citationRefs.forEach(ref => {
                        ref.addEventListener('click', function() {
                            const citationId = this.getAttribute('data-citation');
                            const citation = document.getElementById(citationId);
                            if (citation) {
                                citation.scrollIntoView({ behavior: 'smooth' });
                                citation.classList.add('citation-highlight');
                                setTimeout(() => {
                                    citation.classList.remove('citation-highlight');
                                }, 2000);
                            }
                        });
                    });
                }
                
                // Scroll to bottom of messages
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        }
        
        // Update UI
        setActiveChat(chatId);
        
        // Add current chat class to the history item
        const historyItems = document.querySelectorAll('.history-item');
        historyItems.forEach(item => {
            if (item.dataset.id === chatId) {
                item.id = 'current-chat';
            } else if (item.id === 'current-chat') {
                item.id = '';
            }
        });
        
        console.log("Chat loaded successfully");
    } else {
        console.error("Failed to find chat with ID:", chatId);
    }
}

function getSavedChats() {
    try {
        // Check if localStorage is available
        if (typeof localStorage === 'undefined') {
            console.error('localStorage is not available in this browser');
            return [];
        }
        
        console.log("Attempting to retrieve chat history from localStorage");
        const savedChats = localStorage.getItem('abby_chat_history');
        console.log("Raw data from localStorage:", savedChats ? "Data found (length: " + savedChats.length + ")" : "No data found");
        
        if (!savedChats) {
            console.log("No saved chats found in localStorage");
            return [];
        }
        
        try {
            const parsedChats = JSON.parse(savedChats);
            if (!Array.isArray(parsedChats)) {
                console.error("Parsed chat history is not an array:", typeof parsedChats);
                return [];
            }
            
            console.log("Successfully parsed chat history:", parsedChats.length, "chats found");
            if (parsedChats.length > 0) {
                console.log("First chat:", parsedChats[0].id, parsedChats[0].title);
                console.log("Chat IDs:", parsedChats.map(chat => chat.id));
            }
            return parsedChats;
        } catch (parseError) {
            console.error('Error parsing chat history JSON:', parseError);
            return [];
        }
    } catch (e) {
        console.error('Error loading chat history:', e);
        // Try to recover by clearing corrupted storage
        try {
            localStorage.removeItem('abby_chat_history');
            console.log("Cleared potentially corrupted localStorage data");
        } catch (clearError) {
            console.error('Failed to clear corrupted storage:', clearError);
        }
        return [];
    }
}

function saveChat(chatHistory) {
    try {
        // Check if localStorage is available
        if (typeof localStorage === 'undefined') {
            console.error('localStorage is not available in this browser');
            return;
        }
        
        if (!chatHistory || !chatHistory.id) {
            console.error("Invalid chat history object:", chatHistory);
            return;
        }
        
        // Log chat details
        console.log("Saving chat:", chatHistory.id, chatHistory.title);
        console.log("Chat content length:", chatHistory.messages ? chatHistory.messages.length : 0);
        
        // Get existing chats
        let savedChats = getSavedChats();
        console.log("Retrieved existing chats for saving:", savedChats.length);
        
        // Remove this chat if it already exists
        const existingIndex = savedChats.findIndex(chat => chat.id === chatHistory.id);
        if (existingIndex !== -1) {
            console.log("Updating existing chat in history:", chatHistory.id);
            savedChats.splice(existingIndex, 1);
        }
        
        // Check if the chat has any content before saving
        const messagesHtml = chatHistory.messages || '';
        if (!messagesHtml.includes('message-container')) {
            console.warn("Chat appears to have no messages, skipping save");
            return;
        }
        
        // Add the new chat at the beginning
        savedChats.unshift(chatHistory);
        console.log("Added chat to history. Total chats:", savedChats.length);
        
        // Keep only the last 10 chats to avoid localStorage limits
        if (savedChats.length > 10) {
            savedChats = savedChats.slice(0, 10);
            console.log("Trimmed chat history to 10 items");
        }
        
        // Save back to localStorage
        const chatHistoryJSON = JSON.stringify(savedChats);
        try {
            localStorage.setItem('abby_chat_history', chatHistoryJSON);
            console.log("Successfully saved chat history to localStorage");
            
            // Verify it was saved correctly
            const verifyChats = localStorage.getItem('abby_chat_history');
            if (verifyChats) {
                console.log("Verified save: data exists in localStorage");
                
                // IMPORTANT: Always refresh the history display immediately after saving
                console.log("Refreshing chat history display after save");
                initializeChatHistory();
            } else {
                console.error("Failed to verify localStorage save");
            }
        } catch (storageError) {
            console.error("Error writing to localStorage:", storageError);
            if (storageError.name === 'QuotaExceededError') {
                console.error("Storage quota exceeded - trying to save a smaller version");
                // Try saving just the chat IDs and titles
                const minimalChats = savedChats.map(chat => ({
                    id: chat.id,
                    title: chat.title,
                    timestamp: chat.timestamp
                }));
                localStorage.setItem('abby_chat_history', JSON.stringify(minimalChats));
                
                // Still try to refresh the history display
                initializeChatHistory();
            }
        }
    } catch (e) {
        console.error('Error saving chat history:', e);
    }
}

// Add sidebar styles once
function addSidebarStyles() {
    // Remove any existing style with the ID 'sidebar-styles'
    const existingStyle = document.getElementById('sidebar-styles');
    if (existingStyle) {
        existingStyle.remove();
    }
    
    // Add styles for the history items
    const style = document.createElement('style');
    style.id = 'sidebar-styles';
    style.textContent = `
        .history-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid rgba(0,0,0,0.1);
            transition: background-color 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 14px;
        }
        .history-item:hover {
            background-color: rgba(0,0,0,0.05);
        }
        .history-item.active {
            background-color: rgba(134, 164, 135, 0.2);
            font-weight: bold;
        }
        #chatHistoryContainer {
            max-height: calc(100vh - 150px);
            overflow-y: auto;
            border-top: 1px solid rgba(0,0,0,0.1);
            margin-top: 10px;
        }
    `;
    document.head.appendChild(style);
    console.log("Added sidebar styles");
}

// Helper function to get the current session ID
function getCurrentSessionId() {
    // Use the global conversation ID if available
    if (window.conversationId) {
        return window.conversationId;
    }
    
    // Generate a new ID if not available
    window.conversationId = 'chat_' + Date.now();
    console.log("Created new conversation ID:", window.conversationId);
    return window.conversationId;
}
