/**
 * Quick Exit Functionality
 * Allows users to quickly navigate away from the site for privacy/safety reasons
 */
document.addEventListener('DOMContentLoaded', function() {
    const quickExitButton = document.getElementById('quickExit');
    
    if (quickExitButton) {
        // Handle quick exit button click
        quickExitButton.addEventListener('click', function() {
            performQuickExit();
        });
    }
    
    // Add keyboard shortcut (Single Escape key press triggers exit)
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            performQuickExit();
        }
    });
    
    function performQuickExit() {
        // Create overlay for fade effect
        const fadeOverlay = document.createElement('div');
        fadeOverlay.className = 'exit-overlay';
        fadeOverlay.style.position = 'fixed';
        fadeOverlay.style.top = '0';
        fadeOverlay.style.left = '0';
        fadeOverlay.style.width = '100%';
        fadeOverlay.style.height = '100%';
        fadeOverlay.style.backgroundColor = 'white';
        fadeOverlay.style.zIndex = '9999';
        fadeOverlay.style.opacity = '0';
        fadeOverlay.style.transition = 'opacity 0.3s ease-in-out';
        document.body.appendChild(fadeOverlay);
        
        // Show a brief exit message
        const exitMessage = document.createElement('div');
        exitMessage.style.position = 'absolute';
        exitMessage.style.top = '50%';
        exitMessage.style.left = '50%';
        exitMessage.style.transform = 'translate(-50%, -50%)';
        exitMessage.style.color = '#333';
        exitMessage.style.fontSize = '16px';
        exitMessage.style.fontWeight = '500';
        exitMessage.style.opacity = '0';
        exitMessage.style.transition = 'opacity 0.3s ease-in-out';
        exitMessage.textContent = 'Exiting...';
        fadeOverlay.appendChild(exitMessage);
        
        // Trigger the fade in
        setTimeout(() => {
            fadeOverlay.style.opacity = '1';
            exitMessage.style.opacity = '1';
        }, 10);
        
        // First, clear any locally stored data
        try {
            localStorage.clear();
            sessionStorage.clear();
        } catch (e) {
            console.error('Failed to clear storage', e);
        }
        
        // Navigate to a neutral site after fade effect
        setTimeout(() => {
            window.location.replace('https://www.google.com');
        }, 500);
    }
}); 