// FITUR 3: REAL-TIME TRACKING LOGIC
    const trackingOverlay = document.getElementById('tracking-overlay');
    const startTrackingBtn = document.getElementById('start-tracking-btn');
    let trackingInterval = null;
    let isTracking = false;

    if (startTrackingBtn) {
        startTrackingBtn.addEventListener('click', () => {
            isTracking = !isTracking;
            if (isTracking) {
                startTrackingBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Tracking';
                startTrackingBtn.style.background = '#ff4757';
                
                // Samakan ukuran canvas overlay dengan resolusi video asli
                trackingOverlay.width = video.videoWidth;
                trackingOverlay.height = video.videoHeight;
                const ctx = trackingOverlay.getContext('2d');

                // Kirim frame ke server setiap 300ms (sekitar 3 FPS)
                trackingInterval = setInterval(async () => {
                    // Ambil frame
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    const base64Frame = canvas.toDataURL('image/jpeg', 0.5); // kompres kualitas agar ringan

                    try {
                        // Tembak ke endpoint /predict_live Flask
                        const response = await fetch('/predict_live', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: base64Frame })
                        });
                        const result = await response.json();
                        
                        // Bersihkan canvas
                        ctx.clearRect(0, 0, trackingOverlay.width, trackingOverlay.height);
                        
                        // Gambar Bounding Box (Kotak Wajah)
                        if (result.faces) {
                            result.faces.forEach(face => {
                                const [x, y, w, h] = face.box;
                                
                                // Gambar Kotak Neon
                                ctx.strokeStyle = '#00d2ff';
                                ctx.lineWidth = 3;
                                ctx.strokeRect(x, y, w, h);
                                
                                // Gambar Background Teks
                                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                                ctx.fillRect(x, y - 30, w, 30);
                                
                                // Gambar Teks Emosi & Akurasi
                                ctx.fillStyle = '#00d2ff';
                                ctx.font = 'bold 18px Orbitron';
                                ctx.fillText(`${face.emotion} (${face.confidence}%)`, x + 5, y - 10);
                            });
                        }
                    } catch (err) {
                        console.error("Tracking Error:", err);
                    }
                }, 300);

            } else {
                // Matikan Tracking
                startTrackingBtn.innerHTML = '<i class="fa-solid fa-eye"></i> Live Tracking';
                startTrackingBtn.style.background = 'rgba(255, 255, 255, 0.05)';
                clearInterval(trackingInterval);
                trackingOverlay.getContext('2d').clearRect(0, 0, trackingOverlay.width, trackingOverlay.height);
            }
        });
    }

    // Jangan lupa modifikasi Fungsi stopCamera() agar mematikan interval juga:
    function stopCamera() {
        if (stream) stream.getTracks().forEach(track => track.stop());
        if (trackingInterval) clearInterval(trackingInterval);
        isTracking = false;
        if(startTrackingBtn) startTrackingBtn.innerHTML = '<i class="fa-solid fa-eye"></i> Live Tracking';
    }