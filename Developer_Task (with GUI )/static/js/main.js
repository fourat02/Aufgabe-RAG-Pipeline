let isProcessing = false;

function processPDFs() {
    if (isProcessing) return;

    isProcessing = true;
    document.getElementById('processBtn').disabled = true;
    document.getElementById('processBtn').innerHTML = '<span class="loading"></span> Processing...';
    document.getElementById('progressContainer').style.display = 'block';

    fetch('/process', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showStatus(data.error, 'error');
                resetProcessButton();
            } else {
                checkProcessingStatus();
            }
        })
        .catch(error => {
            showStatus('Error: ' + error.message, 'error');
            resetProcessButton();
        });
}

function checkProcessingStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('progressMessage').textContent = data.message;
            document.getElementById('progressFill').style.width = data.progress + '%';

            if (data.status === 'processing') {
                setTimeout(checkProcessingStatus, 1000);
            } else if (data.status === 'complete') {
                showStatus('✅ ' + data.message, 'success');
                resetProcessButton();
                document.getElementById('processBtn').textContent = '✅ Ready for Questions';
            } else if (data.status === 'error') {
                showStatus('❌ Error: ' + data.message, 'error');
                resetProcessButton();
            }
        })
        .catch(error => {
            showStatus('Status check error: ' + error.message, 'error');
            resetProcessButton();
        });
}

function resetProcessButton() {
    isProcessing = false;
    document.getElementById('processBtn').disabled = false;
    if (document.getElementById('processBtn').textContent.includes('Processing')) {
        document.getElementById('processBtn').textContent = '⚡ Process PDFs';
    }
    setTimeout(() => {
        document.getElementById('progressContainer').style.display = 'none';
    }, 2000);
}

function askQuickQuestion(qId) {
    document.getElementById('answerArea').innerHTML = '<span class="loading"></span> Searching for answer...';

    fetch(`/quick-question/${qId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('answerArea').textContent = '❌ Error: ' + data.error;
            } else {
                displayAnswer(data);
                if (data.question) {
                    document.getElementById('questionInput').value = data.question;
                }
            }
        })
        .catch(error => {
            document.getElementById('answerArea').textContent = '❌ Error: ' + error.message;
        });
}

function askCustomQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) {
        alert('Please enter a question!');
        return;
    }

    document.getElementById('askBtn').disabled = true;
    document.getElementById('askBtn').innerHTML = '<span class="loading"></span> Searching...';
    document.getElementById('answerArea').innerHTML = '<span class="loading"></span> Searching for answer...';

    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('answerArea').textContent = '❌ Error: ' + data.error;
        } else {
            displayAnswer(data);
        }
        document.getElementById('askBtn').disabled = false;
        document.getElementById('askBtn').textContent = '🔍 Ask Question';
    })
    .catch(error => {
        document.getElementById('answerArea').textContent = '❌ Error: ' + error.message;
        document.getElementById('askBtn').disabled = false;
        document.getElementById('askBtn').textContent = '🔍 Ask Question';
    });
}

function displayAnswer(data) {
    let answer = `📝 Answer:\n${data.answer}\n\n`;

    if (data.sources && data.sources.length > 0) {
        answer += `📚 Sources: ${data.sources.join(', ')}\n`;
    }

    if (data.confidence) {
        answer += `🎯 Confidence: ${(data.confidence * 100).toFixed(1)}%\n`;
    }

    document.getElementById('answerArea').textContent = answer;
}

function showStatus(message, type) {
    // You can implement a status notification system here
    console.log(`${type.toUpperCase()}: ${message}`);
}

// Enter key support for question input
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('questionInput').addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            askCustomQuestion();
        }
    });
});