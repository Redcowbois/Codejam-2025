## 💡 Inspiration
All of us on the team have had the same experience: you’re lifting, your form starts drifting, you forget whether you’re on rep 6 or 8, and you cut the set short because no one’s there to motivate you. We wondered why our equipment couldn’t help us with those moments. That’s where SmartBell was born: from wanting a dumbbell that didn’t just sit there, but actually supported us, coached us, and pushed us to be better.

## What It Does

SmartBell brings together sensing, audio feedback, and computer vision to create an intelligent strength-training experience.

Sensors mounted at the base of the dumbbell detect when the weight is resting on the ground. The moment the user lifts the dumbbell, SmartBell automatically starts tracking the set, no buttons, no setup, completely hands-free. Throughout the lift, the system streams real-time progress to the computer dashboard, showing rep count, tempo, and set duration.

As the user begins to approach failure, SmartBell’s onboard speaker provides motivational cues to help them maintain effort and finish strong. This creates an experience similar to having a personal trainer present.

Computer vision plays a major role: it monitors the user’s movement, checks form quality, and identifies which exercise is being performed. Based on the detected exercise, SmartBell automatically logs sets and reps, reducing the need for manual tracking.

## 🛠️ How We Built It

We built SmartBell by combining hardware sensing, embedded systems, computer vision, and a multi-component backend into one cohesive platform.

### 🔌 Embedded Hardware & Sensors  
We started by outfitting the dumbbell with sensors mounted on the bottom to detect when it’s resting on the ground. These sensors feed into an STM32 microcontroller that handles all real-time detection. We originally explored using a load cell but pivoted due to documentation issues, ultimately choosing a more reliable sensing approach.

### 📡 Multi-STM32 Communication System  
To keep the system modular, we used **two STM32 boards**:
- **STM32 #1** collects sensor data and publishes it to an MQTT server.
- **STM32 #2** is dedicated to audio output. When the backend detects that the user is nearing fatigue, it sends an MQTT message back to trigger spoken encouragement.

This separation made each subsystem simpler, but required robust networking and timing coordination.

### 🧠 Computer Vision Pipeline  
A dedicated computer vision module runs on the PC. Using a USB camera, we process live video frames to:
- Recognize which exercise the user is performing  
- Check for basic form cues  
- Provide set and rep counting independent of sensor data  

This CV subsystem runs in parallel with the hardware sensors for improved accuracy and reliability.

### 🖥️ Backend & Integration  
Everything synchronizes through a **Flask-based backend** that handles:
- MQTT communication with both STM32 boards  
- Real-time tracking logic (sets, reps, fatigue estimation)  
- CV inference and data fusion  
- A live dashboard that visualizes workout progress  

We implemented multithreading to manage real-time sensor streams, video processing, and outgoing messages without blocking or delays.

### 🔗 Bringing It All Together  
Finally, we integrated all three pillars—hardware sensing, audio feedback, and computer vision—into one workflow. The result is a system that automatically detects when a set starts, tracks it intelligently, cheers when needed, and logs everything without any manual input from the user.


## ⚠️ Challenges We Ran Into

### 🌐 Networking Challenges  
We built a multi-STM32 system where one microcontroller publishes data to an MQTT server, and another receives server messages to trigger audio encouragement when the user approaches failure. Integrating this with a third subsystem—the computer vision pipeline—introduced a lot of synchronization and communication complexity.

### 🖥️ Backend Challenges  
Our Flask backend required multithreading to handle real-time sensor data, computer vision processing, and outgoing MQTT messages simultaneously. Coordinating these threads reliably was a significant challenge.

### 🔧 Hardware Challenges  
Originally, we planned to use a load cell to detect when the dumbbell was on the ground. However, due to the lack of documentation, we had to pivot to alternative sensing methods and redesign our approach on the fly.

### 🐞 Debugging Complexity  
With multiple subsystems: electrical, micro-controller firmware, drivers, backend software, computer vision, and networking, debugging became especially difficult. A single issue could originate from any layer, and isolating the source often required extensive testing.


## 🏆 Accomplishments We're Proud Of
- We engineered a fully integrated hardware–software system that detects reps, analyzes form, and reacts to user fatigue responsively.  
- We designed a multi-STM32 architecture from scratch and got reliable MQTT communication across all subsystems. 
- We pushed through hardware issues, CV challenges, and debugging across multiple layers to deliver a final demo. 
- Most importantly, we created a tool that genuinely improves the workout experience and reflects the best of each team member’s skills all while having fun!!

## What we learned

## 🚀 What's Next for SmartBell

### 🖨️ 3D-Printed Mount
Replace the current taped attachment with a custom 3D-printed mount. A polished, form-fitting holder will improve durability, safety, and overall user experience.

### 🧠 Custom PCB Design
We're currently using an STM32 Discovery board with many unused peripherals. Designing our own compact PCB would shrink the device, lower cost, and include only the components we actually need.

### 📚 Expanded Exercise Library
Add more exercises beyond the current strength-training set to make SmartBell a more comprehensive workout assistant.

### 🏃‍♂️ Broader Training Support
Since SmartBell already works across different dumbbell weights, we can extend support to other training styles—cardio, pilates, bodyweight movements, and even sport-specific tracking.

