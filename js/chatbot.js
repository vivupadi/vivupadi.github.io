async function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    
    let response = await fetch("http://127.0.0.1:5000/chat", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    });

    let data = await response.json();
    document.getElementById("chat-output").innerHTML += "<p><b>You:</b> " + userInput + "</p>";
    document.getElementById("chat-output").innerHTML += "<p><b>Bot:</b> " + data.reply + "</p>";
}
