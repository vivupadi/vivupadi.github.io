// DOM Elements
const chatOutput = document.getElementById("chat-output");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

// Add event listeners for both button click and Enter key
sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
    const inputText = userInput.value.trim();
    if (!inputText) return;

    // Clear input and disable during processing
    userInput.value = "";
    userInput.disabled = true;
    sendButton.disabled = true;

    // Add user message to chat
    appendMessage("You", inputText);

    try {
        // Show typing indicator
        const typingIndicator = appendMessage("Bot", "Thinking...", true);
        
        // API call with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
        
        const response = await fetch("https://vivupadi-github-io.onrender.com/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: inputText }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Replace typing indicator with actual response
        replaceMessage(typingIndicator, "Bot", data.reply || "I didn't understand that.");
        
    } catch (error) {
        // Handle different error types
        const errorMessage = error.name === "AbortError" 
            ? "Request timed out. Please try again." 
            : `Error: ${error.message}`;
        
        appendMessage("Bot", errorMessage);
        console.error("Chat error:", error);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
        
        // Auto-scroll to bottom
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
}

// Helper functions
function appendMessage(sender, text, isTyping = false) {
    const messageElement = document.createElement("p");
    messageElement.innerHTML = `<b>${sender}:</b> ${text}`;
    if (isTyping) messageElement.id = "typing-indicator";
    chatOutput.appendChild(messageElement);
    return messageElement;
}

function replaceMessage(element, sender, text) {
    element.innerHTML = `<b>${sender}:</b> ${text}`;
    element.id = "";
}