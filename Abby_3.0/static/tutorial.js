document.addEventListener('DOMContentLoaded', function() {
  console.log("Tutorial JS loaded");
  
  // Check if the user has already completed the tutorial OR if force_show_tutorial is set
  const forceShowTutorial = sessionStorage.getItem('force_show_tutorial') === 'true';
  const hasCompletedTutorial = localStorage.getItem('abby_tutorial_completed') === 'true';
  
  // Clear the force show flag
  if (forceShowTutorial) {
    sessionStorage.removeItem('force_show_tutorial');
  }
  
  console.log("Tutorial status:", { forceShowTutorial, hasCompletedTutorial });
  
  // Tutorial steps definition
  const tutorialSteps = [
    {
      title: "Welcome to Abby",
      content: "Abby is your reproductive health assistant. I provide accurate information about reproductive health, abortion clinics, and state policies.",
      icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-10 w-10 text-primary-500"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path></svg>',
      highlight: null
    },
    {
      title: "How to Ask Questions",
      content: "Type your questions in the chat box below. Be specific about your location for location-based information.",
      icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-10 w-10 text-primary-500"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
      example: {
        question: "What are the abortion laws in Texas?",
        description: "Ask about specific states to get detailed policy information"
      },
      highlight: {
        selector: "#userInput",
        position: "bottom"
      }
    },
    {
      title: "Find Nearby Clinics",
      content: "I can show clinics on a map with services information when you ask about locations.<br><br>When you ask about clinics near a specific area, I will show you a map with all available healthcare facilities that provide reproductive health services. The map includes detailed information about each clinic including:<br><br>• Types of services offered<br>• Contact information and website links<br>• Distance from your specified location<br>• Insurance and payment options<br>• Appointment availability<br><br>This feature helps you locate the closest available options for care. You can click on any clinic marker to view more details about that specific facility. The map also includes a list view of all clinics that you can sort by distance or services offered.<br><br>For your privacy, location searches are not stored after your session ends, and you can always clear your search history by starting a new chat.<br><br>Try asking about clinics near major cities or zip codes for the most accurate results.",
      icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-10 w-10 text-primary-500"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
      example: {
        question: "Where can I find abortion clinics near Los Angeles?",
        description: "Ask about a city or zip code to see clinic locations"
      },
      highlight: {
        selector: "#chatMessages",
        position: "top"
      }
    },
    {
      title: "View Information Sources",
      content: "My responses include sources. Click on any citation to expand and see where the information came from.",
      icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-10 w-10 text-primary-500"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6"></path><path d="M16 13H8"></path><path d="M16 17H8"></path><path d="M10 9H8"></path></svg>',
      highlight: null
    },
    {
      title: "Quick Exit Feature",
      content: "If you need privacy, press the ESC key or click the quick exit button in the top right corner to immediately leave this page.",
      icon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-10 w-10 text-primary-500"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>',
      highlight: {
        selector: "#quickExit",
        position: "left"
      }
    }
  ];

  // Initialize tutorial state
  let currentStep = 0;
  let tutorialModal = null;
  let spotlightElement = null;
  
  // Track if we're in the middle of rendering (to prevent multiple clicks)
  let isRendering = false;

  // Create and show the tutorial modal if user hasn't completed it OR if forced to show
  if (!hasCompletedTutorial || forceShowTutorial) {
    console.log("Showing tutorial");
    createTutorialModal();
  } else {
    console.log("Not showing tutorial - already completed");
  }

  // Create the tutorial modal
  function createTutorialModal() {
    try {
      console.log("Creating tutorial modal");
      
      // Clean up any existing tutorial elements first
      if (tutorialModal) {
        tutorialModal.remove();
        tutorialModal = null;
      }
      
      if (spotlightElement) {
        spotlightElement.remove();
        spotlightElement = null;
      }
      
      // Create the modal overlay
      tutorialModal = document.createElement('div');
      tutorialModal.className = 'tutorial-modal-overlay';
      tutorialModal.setAttribute('role', 'dialog');
      tutorialModal.setAttribute('aria-modal', 'true');
      
      // Create the spotlight element
      spotlightElement = document.createElement('div');
      spotlightElement.className = 'tutorial-spotlight';
      spotlightElement.style.display = 'none';
      document.body.appendChild(spotlightElement);

      // Set initial step
      currentStep = 0;
      
      // Render the first step
      renderTutorialStep();
      
      // Add the modal to the document
      document.body.appendChild(tutorialModal);
      
      // Set up escape key handler
      document.addEventListener('keydown', handleEscapeKey);
      
      console.log("Tutorial modal created successfully");
    } catch (error) {
      console.error("Error creating tutorial:", error);
    }
  }

  // Render the current tutorial step
  function renderTutorialStep() {
    try {
      console.log("Rendering step:", currentStep);
      if (isRendering) {
        console.log("Already rendering, skipping");
        return;
      }
      
      isRendering = true;
      
      // Validate current step is in bounds
      if (currentStep < 0 || currentStep >= tutorialSteps.length) {
        console.error("Invalid step index:", currentStep);
        currentStep = 0;
      }
      
      const step = tutorialSteps[currentStep];
      
      // Create modal content
      let modalContent = `
        <div class="tutorial-modal">
          <!-- Close button -->
          <button class="tutorial-close-btn" aria-label="Close tutorial">
            <i class="fas fa-times"></i>
          </button>

          <!-- Header with step indicators and title -->
          <div class="tutorial-header">
            <div class="tutorial-step-indicators">
              ${tutorialSteps.map((_, index) => 
                `<div class="tutorial-step-indicator ${index === currentStep ? 'active' : ''}"></div>`
              ).join('')}
            </div>
            
            <div class="tutorial-title-container">
              <!-- Icon -->
              <div class="tutorial-icon">
                ${step.icon}
              </div>

              <!-- Title -->
              <h3 class="tutorial-title">${step.title}</h3>
            </div>
          </div>

          <!-- Scrollable content area -->
          <div class="tutorial-content">
            <p class="tutorial-text">${step.content}</p>

            ${step.example ? `
              <!-- Example question -->
              <div class="tutorial-example">
                <p class="tutorial-example-description">${step.example.description}</p>
                <div class="tutorial-example-content">
                  <div class="tutorial-example-text">"${step.example.question}"</div>
                  <button class="tutorial-try-btn" data-example="${step.example.question}">Try it</button>
                </div>
              </div>
            ` : ''}
      `;

      // Add visual illustration for the sources step
      if (currentStep === 3) {
        modalContent += `
          <div class="tutorial-illustration">
            <div class="tutorial-illustration-content">
              <div class="tutorial-illustration-text">Information about abortion policies...</div>
              <div class="tutorial-illustration-citation">[1] Source</div>
            </div>
            <div class="tutorial-illustration-detail">
              <p class="tutorial-illustration-citation-title">Citation:</p>
              <p>Guttmacher Institute, "State Facts About Abortion: Texas", 2023</p>
              <a href="#" class="tutorial-illustration-link">View source</a>
            </div>
          </div>
        `;
      }

      // Add quick exit illustration
      if (currentStep === 4) {
        modalContent += `
          <div class="tutorial-illustration">
            <div class="tutorial-illustration-title">Emergency Exit Options:</div>
            <div class="tutorial-exit-options">
              <div class="tutorial-exit-option">
                <span class="tutorial-key">ESC</span> 
                <span>Press once to exit</span>
              </div>
              <div class="tutorial-exit-option">
                <i class="fas fa-sign-out-alt"></i>
                <span>Quick exit button</span>
              </div>
            </div>
          </div>
        `;
      }

      // Add sample questions on last step
      if (currentStep === tutorialSteps.length - 1) {
        modalContent += `
          <div class="tutorial-examples-list">
            <p class="tutorial-examples-title">Try asking Abby:</p>
            <button class="tutorial-example-btn" data-example="What are the abortion laws in Texas?">
              "What are the abortion laws in Texas?"
            </button>
            <button class="tutorial-example-btn" data-example="Where can I find reproductive health clinics in New York?">
              "Where can I find reproductive health clinics in New York?"
            </button>
            <button class="tutorial-example-btn" data-example="What are my contraception options?">
              "What are my contraception options?"
            </button>
          </div>
        `;
      }
      
      // Close content div
      modalContent += `</div>`;

      // Add navigation buttons
      modalContent += `
            <!-- Navigation buttons -->
            <div class="tutorial-nav-buttons">
              <button class="tutorial-skip-btn">Skip Tutorial</button>
              <div class="tutorial-next-buttons">
                ${currentStep > 0 ? `
                  <button class="tutorial-back-btn">
                    <i class="fas fa-arrow-left"></i> Back
                  </button>
                ` : ''}
                <button class="tutorial-next-btn">
                  ${currentStep < tutorialSteps.length - 1 ? 'Next' : 'Get Started'} 
                  <i class="fas fa-arrow-right"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      `;

      // Update the modal contents
      tutorialModal.innerHTML = modalContent;
      
      // Set up button listeners
      setupButtonListeners();
      
      // Update the spotlight
      updateSpotlight();
      
      // Set rendering to false after a short delay
      setTimeout(() => {
        isRendering = false;
      }, 300);
    } catch (error) {
      console.error("Error rendering tutorial step:", error);
      isRendering = false;
    }
  }

  // Setup event listeners for tutorial buttons
  function setupButtonListeners() {
    try {
      console.log("Setting up button listeners for step", currentStep);
      
      // Close button
      const closeBtn = tutorialModal.querySelector('.tutorial-close-btn');
      if (closeBtn) {
        closeBtn.onclick = function(e) {
          if (e) {
            e.preventDefault();
            e.stopPropagation();
          }
          completeTutorial();
        };
      }
      
      // Skip button
      const skipBtn = tutorialModal.querySelector('.tutorial-skip-btn');
      if (skipBtn) {
        skipBtn.onclick = function(e) {
          if (e) {
            e.preventDefault();
            e.stopPropagation();
          }
          completeTutorial();
        };
      }
      
      // Next button - CRITICAL FIX
      const nextBtn = tutorialModal.querySelector('.tutorial-next-btn');
      if (nextBtn) {
        console.log("Adding next button listener");
        nextBtn.onclick = function(e) {
          if (e) {
            e.preventDefault();
            e.stopPropagation();
          }
          
          // Prevent rapid clicking
          if (isRendering) {
            console.log("Still rendering, ignoring click");
            return;
          }
          
          console.log("Next button clicked, current step:", currentStep);
          handleNextStep();
        };
      } else {
        console.error("Next button not found!");
      }
      
      // Back button (if visible)
      const backBtn = tutorialModal.querySelector('.tutorial-back-btn');
      if (backBtn) {
        backBtn.onclick = function(e) {
          if (e) {
            e.preventDefault();
            e.stopPropagation();
          }
          
          // Prevent rapid clicking
          if (isRendering) return;
          
          handlePreviousStep();
        };
      }
      
      // Example buttons (if any)
      const exampleBtns = tutorialModal.querySelectorAll('[data-example]');
      exampleBtns.forEach(btn => {
        btn.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          const exampleText = this.getAttribute('data-example');
          fillChatInput(exampleText);
          // Don't close the tutorial, allow users to explore more
        });
      });
      
    } catch (error) {
      console.error("Error setting up button listeners:", error);
    }
  }

  // Position spotlight on highlighted element
  function updateSpotlight() {
    try {
      const step = tutorialSteps[currentStep];
      
      // Always center the modal by default
      const tutorialModalElement = tutorialModal.querySelector('.tutorial-modal');
      if (tutorialModalElement) {
        // Reset any previous positioning
        tutorialModalElement.style.position = '';
        tutorialModalElement.style.top = '';
        tutorialModalElement.style.left = '';
        tutorialModalElement.style.transform = '';
        
        // Center the modal in the viewport
        tutorialModal.style.display = 'flex';
        tutorialModal.style.justifyContent = 'center';
        tutorialModal.style.alignItems = 'center';
      }
      
      // Handle spotlight if there is a highlighted element
      if (step.highlight && step.highlight.selector) {
        const element = document.querySelector(step.highlight.selector);
        if (element) {
          const rect = element.getBoundingClientRect();
          
          // Show and position the spotlight
          spotlightElement.style.display = 'block';
          spotlightElement.style.top = rect.top + 'px';
          spotlightElement.style.left = rect.left + 'px';
          spotlightElement.style.width = rect.width + 'px';
          spotlightElement.style.height = rect.height + 'px';
        } else {
          // If element not found, hide spotlight
          spotlightElement.style.display = 'none';
        }
      } else {
        // No highlight for this step
        spotlightElement.style.display = 'none';
      }
    } catch (error) {
      console.error("Error updating spotlight:", error);
    }
  }

  // Go to next step - fixed function
  function handleNextStep() {
    console.log("handleNextStep called, current step:", currentStep);
    
    if (currentStep < tutorialSteps.length - 1) {
      // Go to next step
      currentStep++;
      console.log("Moving to next step:", currentStep);
      renderTutorialStep();
    } else {
      // This is the last step
      console.log("This is the last step, completing tutorial");
      completeTutorial();
    }
  }

  // Go to previous step
  function handlePreviousStep() {
    console.log("handlePreviousStep called, current step:", currentStep);
    
    if (currentStep > 0) {
      currentStep--;
      console.log("Moving to previous step:", currentStep);
      renderTutorialStep();
    }
  }

  // Handle escape key press
  function handleEscapeKey(e) {
    if (e && e.key === 'Escape') {
      completeTutorial();
    }
  }

  // Fill the chat input with example text
  function fillChatInput(text) {
    const inputEl = document.getElementById('userInput');
    if (inputEl) {
      inputEl.value = text;
      
      // Enable send button if it exists and was disabled
      const sendButton = document.getElementById('sendButton');
      if (sendButton && sendButton.disabled) {
        sendButton.disabled = false;
      }
      
      // Focus the input
      inputEl.focus();
    }
  }

  // Complete the tutorial
  function completeTutorial() {
    console.log("Completing tutorial");
    
    // Mark tutorial as completed in localStorage
    localStorage.setItem('abby_tutorial_completed', 'true');
    
    // Remove the tutorial elements
    if (tutorialModal) {
      tutorialModal.remove();
      tutorialModal = null;
    }
    
    if (spotlightElement) {
      spotlightElement.remove();
      spotlightElement = null;
    }
    
    // Remove event listener
    document.removeEventListener('keydown', handleEscapeKey);
  }

  // Improved resetTutorial function to ensure tutorial is shown
  window.resetTutorial = function() {
    console.log("Resetting tutorial...");
    
    // Remove tutorial completion flag from localStorage
    localStorage.removeItem('abby_tutorial_completed');
    
    // If we're in the same page, remove existing tutorial elements
    if (tutorialModal) tutorialModal.remove();
    if (spotlightElement) spotlightElement.remove();
    
    // Remove any event listeners
    document.removeEventListener('keydown', handleEscapeKey);
    
    // Explicitly create tutorial now
    currentStep = 0;
    createTutorialModal();
    
    console.log("Tutorial has been reset and should now be visible");
    
    return true;
  };
  
  // Add direct completion function to window for external access
  window.completeTutorial = completeTutorial;
  
  // Add function to immediately show the tutorial without page reload
  window.showTutorial = function() {
    if (!tutorialModal) {
      currentStep = 0;
      createTutorialModal();
      console.log("Tutorial is now visible");
    } else {
      console.log("Tutorial is already visible");
    }
  };
});

// Add CSS to style the tutorial
const tutorialStyles = document.createElement('style');
tutorialStyles.textContent = `
  .tutorial-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  .tutorial-modal {
    position: relative;
    max-width: 400px;
    width: 90%;
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    padding: 24px;
    z-index: 1001;
    margin: 0 auto;
  }
  
  .tutorial-close-btn {
    position: absolute;
    top: 16px;
    right: 16px;
    background: none;
    border: none;
    color: #6B7280;
    font-size: 18px;
    cursor: pointer;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
    z-index: 5;
  }
  
  .tutorial-close-btn:hover {
    background-color: #F3F4F6;
    color: #111827;
  }
  
  .tutorial-step-indicators {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 20px;
  }
  
  .tutorial-step-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #E5E7EB;
  }
  
  .tutorial-step-indicator.active {
    background-color: #6b88c2;
    width: 10px;
    height: 10px;
    margin-top: -1px;
  }
  
  .tutorial-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  
  .tutorial-icon {
    margin-bottom: 16px;
    color: #6b88c2;
  }
  
  .tutorial-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 8px 0;
  }
  
  .tutorial-text {
    font-size: 16px;
    line-height: 1.5;
    color: #4B5563;
    margin: 0 0 20px 0;
  }
  
  .tutorial-example {
    background-color: #F9FAFB;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    width: 100%;
  }
  
  .tutorial-example-description {
    font-size: 14px;
    color: #6B7280;
    margin: 0 0 8px 0;
    text-align: left;
  }
  
  .tutorial-example-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  
  .tutorial-example-text {
    flex: 1;
    font-size: 15px;
    color: #111827;
    text-align: left;
  }
  
  .tutorial-try-btn {
    background-color: #6b88c2;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    cursor: pointer;
    white-space: nowrap;
  }
  
  .tutorial-try-btn:hover {
    background-color: #576ea5;
  }
  
  .tutorial-illustration {
    background-color: #F9FAFB;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    width: 100%;
    text-align: left;
  }
  
  .tutorial-illustration-content {
    border-radius: 6px;
    background-color: #EEF2F9;
    padding: 12px;
    margin-bottom: 12px;
  }
  
  .tutorial-illustration-text {
    font-size: 14px;
    color: #4B5563;
    margin-bottom: 8px;
  }
  
  .tutorial-illustration-citation {
    font-size: 14px;
    color: #3B82F6;
    font-weight: 500;
    cursor: pointer;
  }
  
  .tutorial-illustration-detail {
    background-color: white;
    border-radius: 6px;
    padding: 12px;
    border: 1px solid #E5E7EB;
  }
  
  .tutorial-illustration-citation-title {
    font-size: 14px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 4px 0;
  }
  
  .tutorial-illustration-link {
    font-size: 14px;
    color: #3B82F6;
    display: inline-block;
    margin-top: 8px;
    text-decoration: none;
  }
  
  .tutorial-illustration-title {
    font-size: 15px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 12px;
  }
  
  .tutorial-exit-options {
    display: flex;
    justify-content: space-between;
    gap: 16px;
  }
  
  .tutorial-exit-option {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    background-color: white;
    border-radius: 6px;
    padding: 16px;
    border: 1px solid #E5E7EB;
  }
  
  .tutorial-key {
    background-color: #F3F4F6;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 4px 8px;
    font-family: monospace;
    font-size: 14px;
  }
  
  .tutorial-nav-buttons {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    margin-top: 12px;
  }
  
  .tutorial-next-buttons {
    display: flex;
    gap: 12px;
  }
  
  .tutorial-skip-btn {
    background: none;
    border: none;
    color: #6B7280;
    font-size: 14px;
    cursor: pointer;
    padding: 8px 12px;
  }
  
  .tutorial-skip-btn:hover {
    color: #111827;
    text-decoration: underline;
  }
  
  .tutorial-next-btn {
    background-color: #6b88c2;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .tutorial-next-btn:hover {
    background-color: #576ea5;
  }
  
  .tutorial-back-btn {
    background-color: #F3F4F6;
    color: #4B5563;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .tutorial-back-btn:hover {
    background-color: #E5E7EB;
  }
  
  .tutorial-examples-list {
    width: 100%;
    margin-top: 16px;
    text-align: left;
  }
  
  .tutorial-examples-title {
    font-size: 15px;
    font-weight: 500;
    color: #111827;
    margin-bottom: 8px;
  }
  
  .tutorial-example-btn {
    display: block;
    width: 100%;
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 14px;
    text-align: left;
    margin-bottom: 8px;
    cursor: pointer;
    color: #111827;
    transition: background-color 0.2s;
  }
  
  .tutorial-example-btn:hover {
    background-color: #F3F4F6;
    border-color: #D1D5DB;
  }
  
  .tutorial-spotlight {
    position: absolute;
    pointer-events: none;
    border: 2px solid #6b88c2;
    border-radius: 4px;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
    z-index: 999;
  }
`;

document.head.appendChild(tutorialStyles); 