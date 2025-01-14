
// Add this JavaScript code
const initializePortfolioEffects = () => {
 // Magnetic cursor
 const cursor = document.querySelector('.magnetic-cursor');
 let cursorEnlarged = false;

 document.addEventListener('mousemove', (e) => {
     const { clientX: x, clientY: y } = e;
     cursor.style.transform = `translate(${x - 10}px, ${y - 10}px)`;
     
     // Check if cursor is over interactive elements
     const element = document.elementFromPoint(x, y);
     const isClickable = element?.matches('a, button, .project-card') || 
                        element?.closest('a, button, .project-card');
     
     if (isClickable && !cursorEnlarged) {
         cursor.style.width = '40px';
         cursor.style.height = '40px';
         cursorEnlarged = true;
     } else if (!isClickable && cursorEnlarged) {
         cursor.style.width = '20px';
         cursor.style.height = '20px';
         cursorEnlarged = false;
     }
 });

 // 3D Card Effect
 document.querySelectorAll('.project-card').forEach(card => {
     card.addEventListener('mousemove', (e) => {
         const rect = card.getBoundingClientRect();
         const x = e.clientX - rect.left;
         const y = e.clientY - rect.top;
         
         const centerX = rect.width / 2;
         const centerY = rect.height / 2;
         
         const rotateX = (y - centerY) / 10;
         const rotateY = -(x - centerX) / 10;
         
         card.style.transform = `
             perspective(1000px)
             rotateX(${rotateX}deg)
             rotateY(${rotateY}deg)
             scale3d(1.05, 1.05, 1.05)
         `;
     });

     card.addEventListener('mouseleave', () => {
         card.style.transform = 'none';
     });
 });

 // Typing effect
 const typeText = (element, text, speed = 100) => {
     let i = 0;
     element.textContent = '';
     element.classList.add('typing-text');
     
     const typing = setInterval(() => {
         if (i < text.length) {
             element.textContent += text.charAt(i);
             i++;
         } else {
             clearInterval(typing);
             element.classList.remove('typing-text');
             setTimeout(() => typeText(element, text), 3000); // Repeat
         }
     }, speed);
 };

 // Initialize typing effect for header
 const headerText = document.querySelector('.header-content h1');
 if (headerText) {
     typeText(headerText, headerText.textContent);
 }

 // Scroll-based animations
 const observer = new IntersectionObserver((entries) => {
     entries.forEach(entry => {
         if (entry.isIntersecting) {
             entry.target.classList.add('visible');
             
             // Update nav dots
             const section = entry.target;
             document.querySelectorAll('.nav-dot').forEach(dot => {
                 if (dot.dataset.section === section.id) {
                     dot.classList.add('active');
                 } else {
                     dot.classList.remove('active');
                 }
             });
         }
     });
 }, { threshold: 0.3 });

 // Observe all major sections
 document.querySelectorAll('section, .about-section, .projects-section, .contact-section')
     .forEach(section => {
         section.classList.add('fade-in');
         observer.observe(section);
     });

 // Floating navigation click handlers
 document.querySelectorAll('.nav-dot').forEach(dot => {
     dot.addEventListener('click', () => {
         const section = document.getElementById(dot.dataset.section);
         section.scrollIntoView({ behavior: 'smooth' });
     });
 });
};

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', initializePortfolioEffects);
 