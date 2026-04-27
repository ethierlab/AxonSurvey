function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(function() {
    // Show a brief success message
    const button = event.target;
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    button.style.backgroundColor = '#28a745';
    setTimeout(() => {
      button.textContent = originalText;
      button.style.backgroundColor = '';
    }, 2000);
  }).catch(function(err) {
    console.error('Could not copy text: ', err);
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    const button = event.target;
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    button.style.backgroundColor = '#28a745';
    setTimeout(() => {
      button.textContent = originalText;
      button.style.backgroundColor = '';
    }, 2000);
  });
}

function openImageInNewTab(imageUrl) {
  window.open(imageUrl, '_blank');
}

// File browser functionality for quicktrace
document.addEventListener('DOMContentLoaded', function() {
  const fileBrowser = document.getElementById('file_browser');
  const imagePathInput = document.getElementById('image_path');
  const traceForm = document.getElementById('trace-form');
  const loadingOverlay = document.getElementById('loading-overlay');

  if (fileBrowser) {
    const navigateBtn = document.getElementById('navigate-btn');
    
    // When navigate button is clicked, open file dialog
    if (navigateBtn) {
      navigateBtn.addEventListener('click', function() {
        fileBrowser.click();
      });
    }
    
    fileBrowser.addEventListener('change', function() {
      const file = this.files[0];
      if (file) {
        // Show alert explaining that they need to copy the path manually
        alert('⚠️ File selection does not work for this feature.\n\n' +
              'Please:\n' +
              '1. Navigate to your file in the dialog that just opened\n' +
              '2. Right-click the .tif file in Windows File Explorer\n' +
              '3. Select "Copy as path" (or "Copy file path")\n' +
              '4. Paste the full path into the text field above');
        
        // Clear the file input so it can be used again
        this.value = '';
      }
    });
  }

  if (traceForm) {
    traceForm.addEventListener('submit', function() {
      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
      }
    });
  }

  // Handle action buttons
  document.addEventListener('click', function(event) {
    if (event.target.classList.contains('action-btn')) {
      const action = event.target.getAttribute('data-action');
      
      if (action === 'open-tab') {
        const url = event.target.getAttribute('data-url');
        openImageInNewTab(url);
      } else if (action === 'copy-path') {
        const path = event.target.getAttribute('data-path');
        copyToClipboard(path);
      }
    }
  });
});
