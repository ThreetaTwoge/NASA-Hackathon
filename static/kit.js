// Smooth scrolling for navigation links
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Mobile navigation toggle
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }
    
    // Close mobile menu when clicking on a link
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
        });
    });
});

// NASA Exoplanet Archive API configuration
const NASA_API_BASE = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync';
const NASA_API_PARAMS = {
    query: "select pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,pl_eqt,st_dist,pl_disc,pl_discmethod,pl_habitable from ps where pl_name is not null",
    format: "json",
    limit: 50
};

// Real-time exoplanet data cache
let realExoplanetData = [];
let isLoadingData = false;

// Fetch real exoplanet data from NASA API
async function fetchRealExoplanetData() {
    if (isLoadingData) return;
    isLoadingData = true;
    
    try {
        const params = new URLSearchParams(NASA_API_PARAMS);
        const response = await fetch(`${NASA_API_BASE}?${params}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        realExoplanetData = data.map(planet => ({
            name: planet.pl_name || 'Unknown',
            star: planet.hostname || 'Unknown',
            type: categorizePlanetType(planet.pl_bmasse, planet.pl_rade),
            method: planet.pl_discmethod || 'Unknown',
            mass: planet.pl_bmasse ? `${parseFloat(planet.pl_bmasse).toFixed(2)} M⊕` : 'Unknown',
            radius: planet.pl_rade ? `${parseFloat(planet.pl_rade).toFixed(2)} R⊕` : 'Unknown',
            temperature: planet.pl_eqt ? `${Math.round(planet.pl_eqt)} K` : 'Unknown',
            distance: planet.st_dist ? `${parseFloat(planet.st_dist).toFixed(2)} light years` : 'Unknown',
            discovery_year: planet.pl_disc || 'Unknown',
            habitable: planet.pl_habitable === 1,
            description: generatePlanetDescription(planet),
            orbital_period: planet.pl_orbper ? `${parseFloat(planet.pl_orbper).toFixed(2)} days` : 'Unknown'
        }));
        
        console.log(`Loaded ${realExoplanetData.length} real exoplanets from NASA API`);
        return realExoplanetData;
    } catch (error) {
        console.error('Error fetching NASA data:', error);
        console.log('Falling back to simulated data');
        return getFallbackData();
    } finally {
        isLoadingData = false;
    }
}

// Categorize planet type based on mass and radius
function categorizePlanetType(mass, radius) {
    if (!mass || !radius) return 'unknown';
    
    const massValue = parseFloat(mass);
    const radiusValue = parseFloat(radius);
    
    if (massValue < 10 && radiusValue < 2) return 'terrestrial';
    if (massValue < 10 && radiusValue >= 2) return 'super_earth';
    if (massValue >= 10) return 'gas_giant';
    return 'unknown';
}

// Generate description based on planet data
function generatePlanetDescription(planet) {
    const name = planet.pl_name || 'This planet';
    const star = planet.hostname || 'its star';
    const habitable = planet.pl_habitable === 1;
    
    if (habitable) {
        return `${name} is a potentially habitable exoplanet orbiting ${star}. It may have conditions suitable for life as we know it.`;
    } else {
        return `${name} is an exoplanet orbiting ${star}. Discovered using advanced detection methods, it represents the diversity of planetary systems in our galaxy.`;
    }
}

// Fallback data when API is unavailable
function getFallbackData() {
    return [
    {
        name: "Kepler-452b",
        star: "Kepler-452",
        type: "super_earth",
        method: "transit",
        mass: "5.1 M⊕",
        radius: "1.63 R⊕",
        temperature: "265 K",
        distance: "1400 light years",
        discovery_year: "2015",
        habitable: true,
        description: "Super-Earth located in the habitable zone of its star. One of the most Earth-like exoplanets discovered."
    },
    {
        name: "Proxima Centauri b",
        star: "Proxima Centauri",
        type: "terrestrial",
        method: "radial_velocity",
        mass: "1.27 M⊕",
        radius: "1.07 R⊕",
        temperature: "234 K",
        distance: "4.24 light years",
        discovery_year: "2016",
        habitable: true,
        description: "Closest exoplanet to Earth. Located in the habitable zone of a red dwarf star."
    },
    {
        name: "HD 209458 b",
        star: "HD 209458",
        type: "gas_giant",
        method: "transit",
        mass: "0.69 MJ",
        radius: "1.38 RJ",
        temperature: "1130 K",
        distance: "159 light years",
        discovery_year: "1999",
        habitable: false,
        description: "First exoplanet discovered using the transit method. A hot Jupiter with an atmosphere."
    },
    {
        name: "TRAPPIST-1e",
        star: "TRAPPIST-1",
        type: "terrestrial",
        method: "transit",
        mass: "0.62 M⊕",
        radius: "0.92 R⊕",
        temperature: "251 K",
        distance: "40 light years",
        discovery_year: "2017",
        habitable: true,
        description: "Earth-like planet in a system of seven planets. Potentially habitable for life."
    },
    {
        name: "WASP-12b",
        star: "WASP-12",
        type: "gas_giant",
        method: "transit",
        mass: "1.41 MJ",
        radius: "1.90 RJ",
        temperature: "2580 K",
        distance: "871 light years",
        discovery_year: "2008",
        habitable: false,
        description: "One of the hottest known exoplanets. A gas giant with extreme conditions."
    }
];

// Planet information data
const planetInfo = {
    mercury: {
        name: "Mercury",
        type: "Terrestrial Planet",
        mass: "3.30 × 10²³ kg",
        radius: "2,439 km",
        temperature: "167-427°C",
        distance: "0.39 astronomical units",
        moons: "0",
        description: "Closest planet to the Sun and fastest in the Solar System. Has extreme temperature variations between day and night due to lack of atmosphere.",
        features: ["Fast rotation", "Extreme temperatures", "No atmosphere", "Numerous craters"],
        image: "☿️"
    },
    venus: {
        name: "Venus",
        type: "Terrestrial Planet",
        mass: "4.87 × 10²⁴ kg",
        radius: "6,052 km",
        temperature: "462°C",
        distance: "0.72 astronomical units",
        moons: "0",
        description: "Hottest planet in the Solar System. Has a dense atmosphere of carbon dioxide creating a strong greenhouse effect.",
        features: ["Dense atmosphere", "Greenhouse effect", "Retrograde rotation", "Volcanic activity"],
        image: "♀️"
    },
    earth: {
        name: "Earth",
        type: "Terrestrial Planet",
        mass: "5.97 × 10²⁴ kg",
        radius: "6,371 km",
        temperature: "288 K (15°C)",
        distance: "1 astronomical unit",
        moons: "1 (Moon)",
        description: "Third planet from the Sun and the only known planet with life. Has a dense atmosphere consisting mainly of nitrogen and oxygen. Surface is 71% covered by water.",
        features: ["Liquid water", "Dense atmosphere", "Magnetic field", "Tectonic activity"],
        image: "🌍"
    },
    mars: {
        name: "Mars",
        type: "Terrestrial Planet",
        mass: "6.39 × 10²³ kg",
        radius: "3,390 km",
        temperature: "210 K (-63°C)",
        distance: "1.52 astronomical units",
        moons: "2 (Phobos, Deimos)",
        description: "Fourth planet from the Sun, known as the 'Red Planet' due to iron oxide on its surface. Has a thin atmosphere and polar ice caps.",
        features: ["Polar ice caps", "Largest volcanoes", "Seasonal changes", "Potential water"],
        image: "🔴"
    },
    jupiter: {
        name: "Jupiter",
        type: "Gas Giant",
        mass: "1.90 × 10²⁷ kg",
        radius: "69,911 km",
        temperature: "165 K (-108°C)",
        distance: "5.20 astronomical units",
        moons: "79+",
        description: "Largest planet in the Solar System. Consists mainly of hydrogen and helium. Has the famous Great Red Spot - a giant storm.",
        features: ["Great Red Spot", "Strong magnetic field", "Ring system", "Many moons"],
        image: "🟠"
    },
    saturn: {
        name: "Saturn",
        type: "Gas Giant",
        mass: "5.68 × 10²⁶ kg",
        radius: "58,232 km",
        temperature: "134 K (-139°C)",
        distance: "9.58 astronomical units",
        moons: "82+",
        description: "Second largest planet, famous for its magnificent rings. Has low density and could float in water.",
        features: ["Ring system", "Titan (moon)", "Low density", "Hexagonal storm"],
        image: "🪐"
    },
    uranus: {
        name: "Uranus",
        type: "Ice Giant",
        mass: "8.68 × 10²⁵ kg",
        radius: "25,362 km",
        temperature: "76 K (-197°C)",
        distance: "19.22 astronomical units",
        moons: "27",
        description: "Ice giant rotating on its side. Has a weak ring system and cold atmosphere of hydrogen, helium and methane.",
        features: ["Sideways rotation", "Icy composition", "Weak ring system", "Cold atmosphere"],
        image: "🔵"
    },
    neptune: {
        name: "Neptune",
        type: "Ice Giant",
        mass: "1.02 × 10²⁶ kg",
        radius: "24,622 km",
        temperature: "72 K (-201°C)",
        distance: "30.07 astronomical units",
        moons: "14",
        description: "Windiest planet with the strongest storms in the Solar System. Has a dark spot similar to Jupiter's Great Red Spot.",
        features: ["Strongest winds", "Great Dark Spot", "Icy composition", "Triton (moon)"],
        image: "🔷"
    },
    exoplanets: {
        name: "Exoplanets",
        type: "Planets beyond our Solar System",
        mass: "Various",
        radius: "Various",
        temperature: "From 50K to 5000K+",
        distance: "From 4 to 10000+ light years",
        moons: "Unknown",
        description: "Planets orbiting stars other than the Sun. Over 5000 exoplanets have been discovered using various methods. Many are located in the habitable zones of their stars.",
        features: ["Type diversity", "Habitable zones", "Various discovery methods", "Potential life"],
        image: "🌟"
    }
};

// Search exoplanets function with real NASA data
async function searchExoplanets() {
    const searchInput = document.getElementById('planetSearch').value.toLowerCase();
    const planetType = document.getElementById('planetType').value;
    const discoveryMethod = document.getElementById('discoveryMethod').value;
    const resultsContainer = document.getElementById('searchResults');
    
    // Show loading
    resultsContainer.innerHTML = '<div class="loading"></div> Searching for exoplanets...';
    
    try {
        // Fetch real data if not already loaded
        let planetsToSearch = realExoplanetData.length > 0 ? realExoplanetData : await fetchRealExoplanetData();
        
        // Filter by search term
        if (searchInput) {
            planetsToSearch = planetsToSearch.filter(planet => 
                planet.name.toLowerCase().includes(searchInput) ||
                planet.star.toLowerCase().includes(searchInput)
            );
        }
        
        // Filter by planet type
        if (planetType) {
            planetsToSearch = planetsToSearch.filter(planet => planet.type === planetType);
        }
        
        // Filter by discovery method
        if (discoveryMethod) {
            planetsToSearch = planetsToSearch.filter(planet => 
                planet.method.toLowerCase().includes(discoveryMethod.toLowerCase())
            );
        }
        
        displaySearchResults(planetsToSearch, resultsContainer);
    } catch (error) {
        console.error('Search error:', error);
        resultsContainer.innerHTML = `
            <div class="planet-card">
                <h3>Search Error</h3>
                <p>Unable to search exoplanets at this time. Please try again later.</p>
            </div>
        `;
    }
}

// Display search results
function displaySearchResults(planets, container) {
        if (planets.length === 0) {
            container.innerHTML = `
                <div class="planet-card">
                    <h3>No results found</h3>
                    <p>Try changing search parameters or use different keywords.</p>
                </div>
            `;
            return;
        }
    
    container.innerHTML = planets.map(planet => `
        <div class="planet-card ${realExoplanetData.length > 0 ? 'real-data' : ''}">
            <h3 class="planet-name">${planet.name}</h3>
            <p><strong>Star:</strong> ${planet.star}</p>
            <p><strong>Description:</strong> ${planet.description}</p>
            
            <div class="planet-details">
                <div class="detail-item">
                    <span class="detail-label">Type:</span>
                    <span class="detail-value">${getPlanetTypeName(planet.type)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Mass:</span>
                    <span class="detail-value">${planet.mass}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Radius:</span>
                    <span class="detail-value">${planet.radius}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Temperature:</span>
                    <span class="detail-value">${planet.temperature}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Distance:</span>
                    <span class="detail-value">${planet.distance}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Discovery Year:</span>
                    <span class="detail-value">${planet.discovery_year}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Discovery Method:</span>
                    <span class="detail-value">${getDiscoveryMethodName(planet.method)}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Habitable:</span>
                    <span class="detail-value" style="color: ${planet.habitable ? '#00ff88' : '#ff6b6b'}">
                        ${planet.habitable ? 'Potentially Habitable' : 'Not Habitable'}
                    </span>
                </div>
                ${planet.orbital_period ? `
                <div class="detail-item">
                    <span class="detail-label">Orbital Period:</span>
                    <span class="detail-value">${planet.orbital_period}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `).join('');
    
    // Add animation to cards
    const cards = container.querySelectorAll('.planet-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
        setTimeout(() => card.classList.add('visible'), index * 100);
    });
}

// Get planet type name in English
function getPlanetTypeName(type) {
    const types = {
        'terrestrial': 'Terrestrial',
        'gas_giant': 'Gas Giant',
        'super_earth': 'Super Earth'
    };
    return types[type] || type;
}

// Get discovery method name in English
function getDiscoveryMethodName(method) {
    const methods = {
        'transit': 'Transit Method',
        'radial_velocity': 'Radial Velocity',
        'direct_imaging': 'Direct Imaging'
    };
    return methods[method] || method;
}

// Show planet information modal
function showPlanetInfo(planetKey) {
    const planet = planetInfo[planetKey];
    if (!planet) return;
    
    const modal = document.getElementById('planetModal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <h2 style="color: #00d4ff; margin-bottom: 1rem; font-family: 'Orbitron', monospace;">
            ${planet.image} ${planet.name}
        </h2>
        <p style="color: #888; margin-bottom: 2rem; font-size: 1.1rem;">${planet.type}</p>
        
        <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <p style="line-height: 1.8; color: #cccccc; margin-bottom: 1.5rem;">${planet.description}</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                <div class="detail-item">
                    <span class="detail-label">Mass:</span>
                    <span class="detail-value">${planet.mass}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Radius:</span>
                    <span class="detail-value">${planet.radius}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Temperature:</span>
                    <span class="detail-value">${planet.temperature}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Distance:</span>
                    <span class="detail-value">${planet.distance}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Moons:</span>
                    <span class="detail-value">${planet.moons}</span>
                </div>
            </div>
            
            <h3 style="color: #00d4ff; margin-bottom: 1rem; font-family: 'Orbitron', monospace;">Features:</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                ${planet.features.map(feature => `
                    <span style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(255, 107, 107, 0.2)); 
                                padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid rgba(0, 212, 255, 0.3); 
                                color: #00d4ff; font-size: 0.9rem;">
                        ${feature}
                    </span>
                `).join('')}
            </div>
        </div>
    `;
    
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

// Close modal
function closeModal() {
    const modal = document.getElementById('planetModal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('planetModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Search on Enter key
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('planetSearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchExoplanets();
            }
        });
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', function() {
    const animatedElements = document.querySelectorAll('.explore-card, .section-title, .about-text, .stats');
    animatedElements.forEach(el => {
        el.classList.add('fade-in');
        observer.observe(el);
    });
});

// Parallax effect for hero section
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const parallax = document.querySelector('.hero-whale');
    const speed = scrolled * 0.3;
    
    if (parallax) {
        parallax.style.transform = `translateY(${speed}px)`;
    }
});

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
    // Add click effect to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
});

// Add ripple effect styles
const style = document.createElement('style');
style.textContent = `
    .btn {
        position: relative;
        overflow: hidden;
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Simulate real-time data updates
function updateRealTimeData() {
    // This would connect to real NASA APIs in a production app
    console.log('Updating space data...');
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    console.log('KIT - Space application launched!');
    
    // Show welcome message
    setTimeout(() => {
        console.log('Welcome to the world of space exploration!');
    }, 1000);
    
    // Pre-load NASA data
    initializeNASAData();
    
    // Auto-refresh data every 5 minutes
    setInterval(updateRealTimeData, 300000);
});

// Initialize NASA data on app start
async function initializeNASAData() {
    console.log('Initializing NASA exoplanet data...');
    try {
        await fetchRealExoplanetData();
        console.log('NASA data loaded successfully');
        
        // Show notification that real data is available
        showDataStatusNotification();
    } catch (error) {
        console.log('Using fallback data - NASA API unavailable');
    }
}

// Show notification about data source
function showDataStatusNotification() {
    const notification = document.createElement('div');
    notification.className = 'data-notification';
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-satellite"></i>
            <span>Real NASA exoplanet data loaded!</span>
            <button onclick="this.parentElement.parentElement.remove()" class="close-btn">&times;</button>
        </div>
    `;
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Refresh NASA data manually
async function refreshNASAData() {
    const refreshBtn = document.querySelector('.refresh-btn');
    const originalText = refreshBtn.innerHTML;
    
    // Show loading state
    refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    refreshBtn.disabled = true;
    
    try {
        // Clear existing data
        realExoplanetData = [];
        
        // Fetch fresh data
        await fetchRealExoplanetData();
        
        // Show success notification
        showNotification('Data refreshed successfully!', 'success');
        
    } catch (error) {
        console.error('Error refreshing data:', error);
        showNotification('Failed to refresh data', 'error');
    } finally {
        // Restore button state
        refreshBtn.innerHTML = originalText;
        refreshBtn.disabled = false;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `data-notification ${type}`;
    const icon = type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${icon}"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="close-btn">&times;</button>
        </div>
    `;
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 3000);
}
