async function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    
    let response = await fetch("https://vivupadi-github-io.onrender.com/chat", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    });

    let data = await response.json();
    document.getElementById("chat-output").innerHTML += "<p><b>You:</b> " + userInput + "</p>";
    document.getElementById("chat-output").innerHTML += "<p><b>Bot:</b> " + data.reply + "</p>";
}
