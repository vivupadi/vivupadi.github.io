const sendButton = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatOutput = document.getElementById("chat-output");

sendButton.addEventListener("click", async () => {
    const userMessage = userInput.value;
    if (userMessage.trim() === "") return;

    chatOutput.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;
    const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userMessage })
    });
    const data = await response.json();
    chatOutput.innerHTML += `<p><b>Bot:</b> ${data.reply}</p>`;
    userInput.value = "";
});