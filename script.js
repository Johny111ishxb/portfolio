
      
document.addEventListener('DOMContentLoaded', function() {
const themeToggleBtn = document.querySelector('.theme-toggle-btn');
const shareBtn = document.getElementById('share-btn');
const scanBtn = document.getElementById('scan-button');

// Theme toggle functionality
const savedTheme = localStorage.getItem('theme') || 'light-mode';
document.body.classList.add(savedTheme);

// Check if the button should initially look like dark or light mode
if (savedTheme === 'dark-mode') {
themeToggleBtn.classList.add('dark-mode');
}

themeToggleBtn.addEventListener('click', function() {
if (document.body.classList.contains('light-mode')) {
   // Switch to dark mode
   document.body.classList.remove('light-mode');
   document.body.classList.add('dark-mode');
   localStorage.setItem('theme', 'dark-mode');
   this.classList.add('dark-mode');
} else {
   // Switch to light mode
   document.body.classList.remove('dark-mode');
   document.body.classList.add('light-mode');
   localStorage.setItem('theme', 'light-mode');
   this.classList.remove('dark-mode');
}
});

// Scan button functionality
scanBtn.addEventListener('click', () => {
window.location.href = '/scan';
});

// Share functionality
shareBtn.addEventListener('click', async () => {
const qrImage = document.getElementById('qr-image');

try {
   const response = await fetch(qrImage.src);
   const blob = await response.blob();
   
   if (navigator.share) {
       await navigator.share({
           title: 'QR Code',
           text: 'Check out this QR Code',
           files: [new File([blob], 'qrcode.png', { type: 'image/png' })]
       });
   } else {
       alert('Web Share API not supported in this browser');
   }
} catch (error) {
   console.error('Error sharing:', error);
   alert('Failed to share QR code');
}


});

});
// Add this JavaScript to your main script file
document.addEventListener('DOMContentLoaded', () => {
// First, add the navigation dots to the page
const navHTML = `
<div class="floating-nav">
   <div class="nav-dot" data-section="about"></div>
   <div class="nav-dot" data-section="projects"></div>
   <div class="nav-dot" data-section="contact"></div>
</div>
`;
document.body.insertAdjacentHTML('beforeend', navHTML);

// Get all sections and nav dots
const sections = {
about: document.querySelector('.about-section'),
projects: document.querySelector('.projects-section'),
contact: document.querySelector('.contact-section')
};
const navDots = document.querySelectorAll('.nav-dot');

// Add click handlers to dots
navDots.forEach(dot => {
dot.addEventListener('click', () => {
   const sectionName = dot.getAttribute('data-section');
   sections[sectionName].scrollIntoView({ behavior: 'smooth' });
});
});

// Create intersection observer for sections
const observerOptions = {
root: null,
threshold: 0.5,
rootMargin: '0px'
};

const observerCallback = (entries) => {
entries.forEach(entry => {
   if (entry.isIntersecting) {
       // Find the corresponding dot and activate it
       const sectionId = entry.target.classList.contains('about-section') ? 'about' :
                       entry.target.classList.contains('projects-section') ? 'projects' :
                       'contact';
       
       // Update active state of dots
       navDots.forEach(dot => {
           if (dot.getAttribute('data-section') === sectionId) {
               dot.classList.add('active');
           } else {
               dot.classList.remove('active');
           }
       });
   }
});
};

const observer = new IntersectionObserver(observerCallback, observerOptions);

// Observe all sections
Object.values(sections).forEach(section => {
if (section) {
   observer.observe(section);
}
});
});
function transitionToBio() {
// Create transition overlay
const overlay = document.createElement('div');
overlay.className = 'page-transition';
document.body.appendChild(overlay);

// Trigger animation
setTimeout(() => {
overlay.classList.add('transition-active');
}, 100);

// Navigate to bio page
setTimeout(() => {
window.location.href = 'bio.html';
}, 500);
}




function toggleFaq(element) {
    const item = element.parentElement;
    const wasActive = item.classList.contains('active');
    
    // Close all FAQs
    document.querySelectorAll('.faq-item').forEach(faq => {
        faq.classList.remove('active');
    });
    
    // Open clicked FAQ if it wasn't active
    if (!wasActive) {
        item.classList.add('active');
    }
}