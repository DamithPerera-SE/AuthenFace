const video = document.getElementById('video');
const snapBtn = document.getElementById('snap');
const status = document.getElementById('status');

// Webcam access
navigator.mediaDevices.getUserMedia({video:true})
.then(stream => video.srcObject = stream)
.catch(err => console.error(err));

snapBtn.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video,0,0,640,480);
    const dataURL = canvas.toDataURL('image/jpeg');
    const base64 = dataURL.split(',')[1];

    fetch('http://127.0.0.1:5000/recognize', {
        method: 'POST',
        headers:{ 'Content-Type': 'application/json' },
        body: JSON.stringify({image: base64})
    })
    .then(res => res.json())
    .then(data => {
        status.innerText = "Status: "+data.result;
        const logDiv = document.getElementById('log');
        const p = document.createElement('p');
        p.innerText = `${new Date().toLocaleTimeString()} - ${data.result}`;
        logDiv.prepend(p);
    });
});
