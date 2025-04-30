document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('inputText');
    const enterButton = document.getElementById('enterButton');
    const resultsList = document.getElementById('resultsList');
    const parseTreeImage = document.getElementById('parseTreeImage');
    const fileInput = document.getElementById('fileInput');
    const filePreview = document.getElementById('filePreview');
    
    // Process input when Enter button is clicked
    enterButton.addEventListener('click', function() {
        processInput();
    });
    
    // Also process when pressing Enter in the textarea (with Shift+Enter for new lines)
    inputText.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default behavior (newline)
            processInput();
        }
    });
    
    // Handle file upload
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                filePreview.innerHTML = '';
                
                if (file.type.startsWith('image/')) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.style.maxHeight = '150px';
                    filePreview.appendChild(img);
                } else if (file.type.startsWith('video/')) {
                    const video = document.createElement('video');
                    video.src = e.target.result;
                    video.controls = true;
                    video.style.maxHeight = '150px';
                    filePreview.appendChild(video);
                }
            };
            
            reader.readAsDataURL(file);
        } else {
            filePreview.textContent = `File selected: ${file.name}`;
        }
    });
    
    // Function to process input (text and/or file) and update the UI
    function processInput() {
        const text = inputText.value.trim();
        const file = fileInput.files[0];
        
        if (!text && !file) {
            alert('Please enter some text or select a file first.');
            return;
        }
        
        // Create form data to send to the server
        const formData = new FormData();
        if (text) {
            formData.append('input_text', text);
        }
        
        if (file) {
            formData.append('file', file);
        }
        
        // Show loading state
        resultsList.innerHTML = '<li>Processing...</li>';
        
        // Send the request to the server
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Update the list
            resultsList.innerHTML = '';
            if (data.list_items && data.list_items.length > 0) {
                data.list_items.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    resultsList.appendChild(li);
                });
            } else {
                resultsList.innerHTML = '<li>No results found.</li>';
            }
            
            // Update the image only if specifically requested by the server
            if (data.image_updated) {
                const timestamp = new Date().getTime();
                parseTreeImage.src = `static/images/parse_tree.png?t=${timestamp}`;
            }
            // Note: With image_updated set to false in the server response,
            // this block will not execute, keeping the current image unchanged
            
            // If there's file info, update the display
            if (data.file_info) {
                const fileDisplay = document.createElement('div');
                fileDisplay.innerHTML = `<p>Processed: ${data.file_info.filename}</p>`;
                
                if (data.file_info.content_type.startsWith('image/')) {
                    fileDisplay.innerHTML += `<img src="${data.file_info.path}" style="max-height: 150px;">`;
                } else if (data.file_info.content_type.startsWith('video/')) {
                    fileDisplay.innerHTML += `<video src="${data.file_info.path}" controls style="max-height: 150px;"></video>`;
                }
                
                // Clear and update the file preview
                filePreview.innerHTML = '';
                filePreview.appendChild(fileDisplay);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultsList.innerHTML = '<li>Error processing input. Please try again.</li>';
        });
    }
});