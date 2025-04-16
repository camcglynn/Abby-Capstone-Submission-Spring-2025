// Global variables
let map = null;
let markers = [];
let placesService = null;
let mapReady = false;

// Check if Google Maps API is ready
function isGoogleMapsReady() {
    return typeof window.google !== 'undefined' && 
           typeof window.google.maps !== 'undefined' && 
           typeof window.google.maps.places !== 'undefined';
}

// Initialize map
function initMap() {
    const mapElement = document.getElementById('clinic-map');
    if (!mapElement) return;
    
    // Check if Google Maps is loaded
    if (!isGoogleMapsReady()) {
        console.error("Google Maps API is not loaded yet");
        return;
    }
    
    map = new google.maps.Map(mapElement, {
        center: { lat: 37.7749, lng: -122.4194 }, // Default to San Francisco
        zoom: 12,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false
    });
    
    // Initialize Places service
    if (window.google && window.google.maps && window.google.maps.places) {
        placesService = new google.maps.places.PlacesService(map);
        mapReady = true;
        console.log("Map initialized successfully");
    }
}

// Listen for Google Maps initialization
document.addEventListener('google-maps-initialized', function() {
    console.log("Received google-maps-initialized event");
    mapReady = true;
});

// Expose the maps API
window.mapsApi = {
    showClinicMap: function(location, mapId) {
        console.log("showClinicMap called with location:", location, "mapId:", mapId);
        
        try {
            // Get the map element using the provided ID
            const mapElement = document.getElementById(mapId);
            if (!mapElement) {
                console.error("Map element not found with ID:", mapId);
                return;
            }
            
            // Show loading indicator
            mapElement.innerHTML = '<div style="text-align:center; padding:20px;"><p>Loading clinic map...</p></div>';
            
            // Format the location for better geocoding - if it's a ZIP code, add USA
            let formattedLocation = location;
            if (/^\d{5}$/.test(location)) {
                formattedLocation = location + ", USA";
                console.log(`Formatted ZIP code to: ${formattedLocation}`);
            }
            
            // Now geocode the location to get proper coordinates
            const geocoder = new google.maps.Geocoder();
            
            geocoder.geocode({
                address: formattedLocation,
                componentRestrictions: {
                    country: 'US'
                }
            }, function(results, status) {
                if (status === "OK" && results && results.length > 0) {
                    // We have coordinates! Create the map
                    const locationLatLng = results[0].geometry.location;
                    console.log(`Geocoded "${formattedLocation}" to:`, locationLatLng.lat(), locationLatLng.lng());
                    console.log(`Full address: ${results[0].formatted_address}`);
                    
                    // Create map
                    const map = new google.maps.Map(mapElement, {
                        center: locationLatLng,
                        zoom: 12,
                        mapTypeControl: false,
                        streetViewControl: false,
                        fullscreenControl: false
                    });
                    
                    // Add marker for user's location
                    new google.maps.Marker({
                        position: locationLatLng,
                        map: map,
                        title: `Your Location: ${results[0].formatted_address}`,
                        icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: 10,
                            fillColor: '#4285F4',
                            fillOpacity: 1,
                            strokeColor: '#ffffff',
                            strokeWeight: 2
                        }
                    });
                    
                    // Initialize Places service
                    const placesService = new google.maps.places.PlacesService(map);
                    
                    // Try different search approaches with terms that work better with Places API
                    tryMultipleSearchMethods(placesService, locationLatLng, map, results[0].formatted_address);
                } else {
                    console.error(`Geocoding failed with status: ${status}`);
                    // Fall back to synthetic data since geocoding failed
                    displaySyntheticData(mapElement, location);
                }
            });
        } catch (error) {
            console.error("Error in showClinicMap:", error);
            try {
                const mapElement = document.getElementById(mapId);
                if (mapElement) {
                    mapElement.innerHTML = getFallbackContent(location);
                }
            } catch (finalError) {
                console.error("Could not display fallback content:", finalError);
            }
        }
    },
    
    displayMapError: function(element, message) {
        if (element) {
        element.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                <p>${message}</p>
            </div>
        `;
        }
    },
    
    // Expose the getFallbackContent function
    getFallbackContent: function(location) {
        return getFallbackContent(location);
    },
    
    // Function to manually display specific clinics in the UI
    // This is a fallback if the Google Maps API doesn't work
    displayStaticClinicList: function(zipCode) {
        // Generate bot message with clinic listings
        const clinicMessage = `
            <h3>Clinics Near ${zipCode}</h3>
            
            <div class="clinic-listing">
                <div class="clinic-name">Planned Parenthood - San Jose</div>
                <div class="clinic-address">1691 The Alameda, San Jose, CA 95126</div>
                <div class="clinic-distance">2.8 miles • Open today: 9:00am - 5:00pm</div>
            </div>
            
            <div class="clinic-listing">
                <div class="clinic-name">Women's Community Clinic</div>
                <div class="clinic-address">2520 Samaritan Dr, San Jose, CA 95124</div>
                <div class="clinic-distance">3.5 miles • Open today: 8:30am - 4:30pm</div>
            </div>
            
            <div id="clinic-map-placeholder">
                <div class="map-placeholder">
                    <div><i class="fas fa-map-marker-alt" style="font-size: 24px; margin-bottom: 12px; color: #6b88c2;"></i></div>
                    <div>Interactive map would appear here</div>
                </div>
            </div>
        `;
        
        // Find the bot's most recent message and update it
        const botMessages = document.querySelectorAll('.bot-message');
        const latestBotMessage = botMessages[botMessages.length - 1];
        
        if (latestBotMessage) {
            latestBotMessage.innerHTML = clinicMessage;
        } else {
            // If no bot message exists, create a new one
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container';
            
            const messageEl = document.createElement('div');
            messageEl.className = 'message bot-message';
            messageEl.innerHTML = clinicMessage;
            
            messageContainer.appendChild(messageEl);
            document.getElementById('chatMessages').appendChild(messageContainer);
        }
    }
};

// Function to add clinics to the map
function addClinicsToMap(location) {
    if (!map || !placesService) {
        console.error("Map or Places service not initialized");
        return;
    }

    // Clear existing markers
    clearMarkers();
    
    // Geocode the location
    const geocoder = new google.maps.Geocoder();
    geocoder.geocode({ address: location }, (results, status) => {
        if (status === 'OK' && results[0]) {
            const locationLatLng = results[0].geometry.location;
            map.setCenter(locationLatLng);
            
            // Search for clinics
            const request = {
                location: locationLatLng,
                radius: '50000', // 50km radius
                type: ['health'],
                keyword: 'abortion clinic'
            };
            
            placesService.nearbySearch(request, (results, status) => {
                if (status === google.maps.places.PlacesServiceStatus.OK) {
                    console.log(`Found ${results.length} clinics`);
                    
                    results.forEach(place => {
                        // Create marker
        const marker = new google.maps.Marker({
            map: map,
                            position: place.geometry.location,
                            title: place.name
                        });
                        
                        markers.push(marker);
                        
                        // Create info window
        const infoWindow = new google.maps.InfoWindow({
                            content: `
                                <div class="info-window">
                                    <h3>${place.name}</h3>
                                    <p>${place.vicinity}</p>
                                    ${place.rating ? `<p>Rating: ${place.rating} ⭐</p>` : ''}
                                    ${place.opening_hours?.open_now ? 
                                        '<p>Status: Open now</p>' : 
                                        place.opening_hours ? '<p>Status: Closed</p>' : ''}
                                </div>
                            `
                        });
                        
                        // Add click listener
        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });
                    });
                    
                    // Fit bounds to show all markers
                    if (markers.length > 0) {
                        const bounds = new google.maps.LatLngBounds();
                        markers.forEach(marker => bounds.extend(marker.getPosition()));
        map.fitBounds(bounds);
                    }
                } else {
                    console.error("Places search failed:", status);
                }
            });
        } else {
            console.error("Geocoding failed:", status);
        }
    });
}

// Helper function to clear markers
function clearMarkers() {
    markers.forEach(marker => marker.setMap(null));
    markers = [];
}

// Create list of clinic data by region
const CLINIC_DATA_BY_REGION = {
    'san_francisco': [
        {
            name: "Planned Parenthood - San Francisco Health Center",
            address: "1650 Valencia St, San Francisco, CA 94110",
            distance: "1.4",
            phone: "(415) 821-1282"
        },
        {
            name: "Women's Options Center at SFGH",
            address: "1001 Potrero Ave, San Francisco, CA 94110",
            distance: "2.3",
            phone: "(415) 206-8476"
        },
        {
            name: "Planned Parenthood - Mission",
            address: "2430 Mission St, San Francisco, CA 94110",
            distance: "3.1",
            phone: "(415) 821-1282"
        }
    ],
    'los_angeles': [
        {
            name: "Planned Parenthood - Beverly Hills",
            address: "8231 W Beverly Blvd, Los Angeles, CA 90048",
            distance: "5.2",
            phone: "(800) 576-5544"
        },
        {
            name: "Family Planning Associates",
            address: "601 S Westmoreland Ave, Los Angeles, CA 90005",
            distance: "8.7",
            phone: "(213) 738-7283"
        },
        {
            name: "Women's Health Specialists",
            address: "3233 W 6th St, Los Angeles, CA 90020",
            distance: "9.1",
            phone: "(213) 386-2606"
        }
    ],
    'new_york': [
        {
            name: "Planned Parenthood - Margaret Sanger Center",
            address: "26 Bleecker St, New York, NY 10012",
            distance: "1.8",
            phone: "(212) 965-7000"
        },
        {
            name: "Planned Parenthood - Manhattan Health Center",
            address: "21 E 22nd St, New York, NY 10010",
            distance: "3.2",
            phone: "(212) 271-0200"
        },
        {
            name: "Midtown Medical Pavilion",
            address: "380 2nd Ave, New York, NY 10010",
            distance: "4.5",
            phone: "(212) 683-7200"
        }
    ],
    'default': [
        {
            name: "Planned Parenthood Health Center",
            address: "123 Main St, Nearest City, CA",
            distance: "3.5",
            phone: "(800) 230-PLAN"
        },
        {
            name: "Family Planning Associates",
            address: "456 Health Avenue, Nearest City, CA",
            distance: "5.1",
            phone: "(800) 435-8742"
        },
        {
            name: "Women's Health Center",
            address: "789 Medical Drive, Nearest City, CA",
            distance: "7.2",
            phone: "(888) 429-7600"
        }
    ]
};

// Function to determine which region's data to use
function getClinicDataForLocation(location) {
    if (!location) return CLINIC_DATA_BY_REGION.default;
    
    const locationLower = location.toLowerCase();
    
    // San Francisco Bay Area
    if (locationLower.includes('san francisco') || locationLower.includes('sf') || 
        locationLower.includes('oakland') || locationLower.includes('berkeley') ||
        locationLower.match(/\b94\d{3}\b/)) {
        return CLINIC_DATA_BY_REGION.san_francisco;
    }
    
    // Los Angeles area
    if (locationLower.includes('los angeles') || locationLower.includes('la') || 
        locationLower.includes('beverly') || locationLower.includes('hollywood') ||
        locationLower.includes('santa monica') || locationLower.includes('pasadena') ||
        locationLower.match(/\b90\d{3}\b/)) {
        return CLINIC_DATA_BY_REGION.los_angeles;
    }
    
    // New York area
    if (locationLower.includes('new york') || locationLower.includes('nyc') || 
        locationLower.includes('manhattan') || locationLower.includes('brooklyn') ||
        locationLower.includes('queens') || locationLower.includes('bronx') ||
        locationLower.match(/\b10\d{3}\b/) || locationLower.match(/\b11\d{3}\b/)) {
        return CLINIC_DATA_BY_REGION.new_york;
    }
    
    // Check for other major cities - Generic handling with city-specific names
    const cities = {
        'chicago': {
            name: "Planned Parenthood - Near North Health Center",
            address: "1200 N LaSalle Dr, Chicago, IL 60610",
            distance: "2.3",
            phone: "(312) 266-1033"
        },
        'detroit': {
            name: "Planned Parenthood - Detroit Health Center",
            address: "4229 Cass Ave, Detroit, MI 48201",
            distance: "1.5",
            phone: "(313) 833-7080"
        },
        'boston': {
            name: "Planned Parenthood - Greater Boston Health Center",
            address: "1055 Commonwealth Ave, Boston, MA 02215",
            distance: "1.9", 
            phone: "(617) 616-1600"
        },
        'seattle': {
            name: "Planned Parenthood - Seattle Health Center",
            address: "2001 E Madison St, Seattle, WA 98122",
            distance: "2.1",
            phone: "(206) 320-7610"
        },
        'atlanta': {
            name: "Feminist Women's Health Center",
            address: "1924 Cliff Valley Way NE, Atlanta, GA 30329",
            distance: "3.7",
            phone: "(404) 728-7900"
        },
        'denver': {
            name: "Planned Parenthood - Denver Health Center",
            address: "921 E 14th Ave, Denver, CO 80218",
            distance: "1.8",
            phone: "(303) 832-5069"
        },
        'miami': {
            name: "Planned Parenthood - Miami Health Center",
            address: "4010 N Kendall Dr, Miami, FL 33176",
            distance: "2.2",
            phone: "(305) 441-2022"
        },
        'phoenix': {
            name: "Planned Parenthood - Phoenix Health Center",
            address: "5651 N 7th St, Phoenix, AZ 85014",
            distance: "2.4",
            phone: "(602) 277-7526"
        }
    };
    
    // Check if the location contains any of these city names
    for (const city in cities) {
        if (locationLower.includes(city)) {
            // Create a custom array with the city-specific clinic first
            return [
                cities[city],
                {
                    name: "Family Planning Associates",
                    address: `123 Main St, ${city.charAt(0).toUpperCase() + city.slice(1)}, USA`,
                    distance: "4.2",
                    phone: "(800) 435-8742"
                },
                {
                    name: "Women's Health Center",
                    address: `456 Health Ave, ${city.charAt(0).toUpperCase() + city.slice(1)}, USA`,
                    distance: "6.1",
                    phone: "(888) 429-7600"
                }
            ];
        }
    }
    
    // Default to generic clinics if no specific location match
    return CLINIC_DATA_BY_REGION.default;
}

// Display clinic information in the chat interface
function displayClinicsInChat(locations) {
    if (!locations || locations.length === 0) return;
    
    // Format the clinic information for display
    let clinicHTML = `<h3>Clinics Near ${locations[0].zipCode || ''}</h3>`;
    
    locations.slice(0, 5).forEach(clinic => {
        // Format hours
        const hours = clinic.hours || '9:00am - 5:00pm';
        
        clinicHTML += `
            <div class="clinic-listing">
                <div class="clinic-name">${clinic.name}</div>
                <div class="clinic-address">${clinic.address}</div>
                <div class="clinic-distance">${formatDistance(clinic.distance)} • Open today: ${hours}</div>
            </div>
        `;
    });
    
    // Find the bot's most recent message and update it
    const botMessages = document.querySelectorAll('.bot-message');
    const latestBotMessage = botMessages[botMessages.length - 1];
    
    if (latestBotMessage) {
        // Check if the message has specific clinic content we should replace
        const hasClinicHeader = latestBotMessage.querySelector('h3') && 
                                latestBotMessage.querySelector('h3').textContent.includes('Clinics');
        
        if (hasClinicHeader) {
            // Replace the entire content
            latestBotMessage.innerHTML = clinicHTML;
        } else {
            // Append to the existing message
            const existingContent = latestBotMessage.innerHTML;
            latestBotMessage.innerHTML = existingContent + '<br><br>' + clinicHTML;
        }
    }
}

// Format distance for display
function formatDistance(distance) {
    if (!distance && distance !== 0) return '';
    
    // Convert to number if it's a string
    const distVal = typeof distance === 'string' ? parseFloat(distance) : distance;
    
    // Format to one decimal place for values with decimals
    return Number.isInteger(distVal) ? 
        `${distVal} miles` : 
        `${distVal.toFixed(1)} miles`;
}

// Fetch clinic data from the Google Maps Places API
function fetchClinicData(location) {
    console.log(`Fetching clinic data for location: ${location}`);
    
    // Log browser details and API status
    console.log("Browser details:", navigator.userAgent);
    console.log("Google Maps API status:", {
        maps: !!window.google?.maps,
        places: !!window.google?.maps?.places,
        geometry: !!window.google?.maps?.geometry,
        geocoder: !!window.google?.maps?.Geocoder
    });
    
    // Check if Google Maps API is available
    if (!window.google || !window.google.maps) {
        console.error("Google Maps API not available. Displaying static clinic data.");
        displayStaticClinicData(location);
        return;
    }
    
    // Show loading indicator in map container
    showMapLoadingIndicator(location);
    
    // Use geocoder to find coordinates
    try {
        console.log(`Using geocoder for location: ${location}`);
        
        // Create a geocoder instance
        const geocoder = new google.maps.Geocoder();
        
        // Geocode the location to get coordinates
        geocoder.geocode({ 'address': location }, function(results, status) {
            console.log(`Geocoding status: ${status}`);
            
            if (status !== 'OK' || !results || results.length === 0) {
                console.error(`Geocoding failed with status: ${status}`);
                // Try direct coordinates for well-known cities
                tryDirectCoordinates(location);
                return;
            }
            
            console.log(`Geocoding successful for ${location}`, results[0].geometry.location);
            
            // Set up request for nearby search
            const request = {
                location: results[0].geometry.location,
                radius: '25000',  // 25km radius
                keyword: 'reproductive health clinic planned parenthood abortion',
                type: ['health']  // Focus on health facilities
            };
            
            try {
                // Create PlacesService instance
                const mapElement = document.getElementById('clinic-map');
                if (!mapElement) {
                    throw new Error("Map element not found");
                }
                
                const placesService = new google.maps.places.PlacesService(mapElement);
                
                console.log("Performing Places API search with request:", request);
                
                // Perform nearby search
                placesService.nearbySearch(request, function(results, status) {
                    console.log(`Places API nearby search status: ${status}`);
                    
                    if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
                        console.log(`Found ${results.length} clinics via nearby search`);
                        displayClinicResults(results, results[0].geometry.location);
                    } else {
                        console.log(`No results from nearby search or error (${status}), trying text search`);
                        
                        // Try text search as a fallback
                        const textSearchRequest = {
                            query: `reproductive health clinics near ${location}`,
                            type: ['health']
                        };
                        
                        placesService.textSearch(textSearchRequest, function(textResults, textStatus) {
                            console.log(`Places API text search status: ${textStatus}`);
                            
                            if (textStatus === google.maps.places.PlacesServiceStatus.OK && textResults && textResults.length > 0) {
                                console.log(`Found ${textResults.length} clinics via text search`);
                                displayClinicResults(textResults, textResults[0].geometry.location);
                            } else {
                                console.error(`Text search failed with status: ${textStatus}`);
                                displayStaticClinicData(location);
                            }
                        });
                    }
                });
            } catch (placesError) {
                console.error(`Error with Places API: ${placesError}`);
                displayStaticClinicData(location);
            }
        });
    } catch (error) {
        console.error(`Error in fetchClinicData: ${error}`);
        displayStaticClinicData(location);
    }
}

// Show loading indicator in map container
function showMapLoadingIndicator(location) {
    // Get the map container and make it visible
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
        mapContainer.style.display = 'block';
        mapContainer.removeAttribute('aria-hidden');
    }
    
    // Show loading indicator in map element
    const mapElement = document.getElementById('clinic-map');
    if (mapElement) {
        mapElement.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px;">
                <div><i class="fas fa-circle-notch fa-spin" style="font-size: 32px; margin-bottom: 16px; color: #2c4c8c;"></i></div>
                <div>Searching for clinics near ${location}...</div>
            </div>
        `;
    }
}

// Try direct coordinates for well-known cities
function tryDirectCoordinates(location) {
    console.log(`Trying direct coordinates for ${location}`);
    
    // Map of city names to coordinates
    const cityCoordinates = {
        'detroit': { lat: 42.3314, lng: -83.0458 },
        'new york': { lat: 40.7128, lng: -74.0060 },
        'los angeles': { lat: 34.0522, lng: -118.2437 },
        'chicago': { lat: 41.8781, lng: -87.6298 },
        'houston': { lat: 29.7604, lng: -95.3698 },
        'philadelphia': { lat: 39.9526, lng: -75.1652 },
        'phoenix': { lat: 33.4484, lng: -112.0740 },
        'san antonio': { lat: 29.4241, lng: -98.4936 },
        'san diego': { lat: 32.7157, lng: -117.1611 },
        'dallas': { lat: 32.7767, lng: -96.7970 },
        'san francisco': { lat: 37.7749, lng: -122.4194 },
        'seattle': { lat: 47.6062, lng: -122.3321 },
        'boston': { lat: 42.3601, lng: -71.0589 },
        'atlanta': { lat: 33.7490, lng: -84.3880 },
        'miami': { lat: 25.7617, lng: -80.1918 }
    };
    
    // Find a matching city for direct coordinates
    const locationLower = location.toLowerCase();
    let coords = null;
    
    for (const city in cityCoordinates) {
        if (locationLower.includes(city)) {
            coords = cityCoordinates[city];
            console.log(`Found direct coordinates for ${city}:`, coords);
            break;
        }
    }
    
    if (!coords) {
        console.log("No direct coordinates found, using static data");
        displayStaticClinicData(location);
        return;
    }
    
    try {
        // Set up direct request for nearby search using coordinates
        const request = {
            location: coords,
            radius: '50000',  // 50km radius
            type: ['health'], // Restrict to health facilities
            keyword: 'reproductive health clinic planned parenthood'
        };
        
        // Create PlacesService instance
        const placesService = new google.maps.places.PlacesService(document.getElementById('clinic-map'));
        
        console.log(`Performing direct Places API search for ${location}`);
        
        // Perform nearby search
        placesService.nearbySearch(request, function(results, status) {
            console.log(`Places API status for direct search: ${status}`);
            
            if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
                console.log(`Found ${results.length} clinics in ${location} area via direct coordinates`);
                displayClinicResults(results, coords);
            } else {
                console.error(`Error with direct search: ${status}, trying text search`);
                
                // Try text search as another option
                placesService.textSearch({
                    query: `reproductive health clinics in ${location}`,
                    location: coords,
                    radius: '50000'
                }, function(textResults, textStatus) {
                    console.log(`Text search status for ${location}: ${textStatus}`);
                    
                    if (textStatus === google.maps.places.PlacesServiceStatus.OK && textResults && textResults.length > 0) {
                        console.log(`Found ${textResults.length} clinics via text search`);
                        displayClinicResults(textResults, coords);
                    } else {
                        console.error(`Text search failed: ${textStatus}, using static data`);
                        displayStaticClinicData(location);
                    }
                });
            }
        });
    } catch (error) {
        console.error(`Error with direct search: ${error}`);
        displayStaticClinicData(location);
    }
}

// Get additional place details when available
function getPlaceDetails(place, placesService, index) {
    return new Promise((resolve, reject) => {
        if (!place || !place.place_id) {
            console.log(`Missing place_id for place at index ${index}`);
            resolve(place); // Return the original place
            return;
        }
        
        console.log(`Fetching details for place: ${place.name}`);
        
        placesService.getDetails(
            { placeId: place.place_id, fields: ['formatted_phone_number', 'website', 'opening_hours', 'rating'] },
            (detailedPlace, status) => {
                if (status === google.maps.places.PlacesServiceStatus.OK) {
                    console.log(`Got details for place ${place.name}:`, detailedPlace);
                    
                    // Merge the details with the original place
                    const mergedPlace = {
                        ...place,
                        formatted_phone_number: detailedPlace.formatted_phone_number || place.formatted_phone_number,
                        website: detailedPlace.website || place.website,
                        opening_hours: detailedPlace.opening_hours || place.opening_hours,
                        rating: detailedPlace.rating || place.rating
                    };
                    
                    resolve(mergedPlace);
                } else {
                    console.warn(`Failed to get details for place ${place.name}: ${status}`);
                    resolve(place); // Return the original place
                }
            }
        );
    });
}

// Display the clinic results - reliable with error handling
async function displayClinicResults(places, center) {
    console.log("Displaying clinic results:", places.length);
    
    // Get the map element
    const mapElement = document.getElementById('clinic-map');
    if (!mapElement) {
        console.error("Map element not found");
        return;
    }
    
    // Build HTML for clinic results
    let html = `
        <div class="static-clinic-list">
            <h3>Reproductive Health Clinics</h3>
            <p class="note">Showing ${places.length} locations. Select a clinic for more information.</p>
    `;
    
    // Process each place
    places.forEach((place, index) => {
        // Calculate distance from center (if coordinates available)
        let distance = "";
        if (center && place.geometry && place.geometry.location) {
            // Calculate distance in kilometers, then convert to miles
            const distanceInKm = google.maps.geometry.spherical.computeDistanceBetween(
                center, 
                place.geometry.location
            ) / 1000; // Convert meters to kilometers
            
            // Convert to miles (1 km ≈ 0.621371 miles)
            const distanceInMiles = (distanceInKm * 0.621371).toFixed(1);
            distance = `${distanceInMiles} miles away`;
        }
        
        // Get the place details for phone number
        let phone = "";
        if (place.formatted_phone_number) {
            phone = place.formatted_phone_number;
        }
        
        // Build the clinic item HTML
        html += `
            <div class="clinic-item" data-place-id="${place.place_id}">
                <h4>${place.name}</h4>
                <p class="address">${place.vicinity || place.formatted_address || 'Address unavailable'}</p>
                ${distance ? `<p class="distance">${distance}</p>` : ''}
                ${phone ? `<p class="phone">Phone: ${phone}</p>` : ''}
            </div>
        `;
    });
    
    html += `
        </div>
    `;
    
    // Find the bot's most recent message and update it
    const botMessages = document.querySelectorAll('.bot-message');
    const latestBotMessage = botMessages[botMessages.length - 1];
    
    if (latestBotMessage) {
        // Check if the message has specific clinic content we should replace
        const hasClinicHeader = latestBotMessage.querySelector('h3') && 
                                latestBotMessage.querySelector('h3').textContent.includes('Clinics');
        
        if (hasClinicHeader) {
            // Replace the entire content
            latestBotMessage.innerHTML = html;
        } else {
            // Append to the existing message
            const existingContent = latestBotMessage.innerHTML;
            latestBotMessage.innerHTML = existingContent + '<br><br>' + html;
        }
    }
}

// Function to get fallback content when Maps API fails
function getFallbackContent(location) {
    return `
        <div style="padding: 20px; text-align: center;">
            <h3>Abortion Clinics Near ${location}</h3>
            <p>We're having trouble loading the map right now. Please try again later or refer to these resources:</p>
            <ul style="text-align: left; display: inline-block;">
                <li><a href="https://www.plannedparenthood.org/health-center" target="_blank">Planned Parenthood Health Center Finder</a></li>
                <li><a href="https://prochoice.org/patients/find-a-provider/" target="_blank">National Abortion Federation Provider Finder</a></li>
                <li><a href="https://www.abortionfinder.org/" target="_blank">Abortion Finder</a></li>
                <li><a href="https://ineedana.com/" target="_blank">I Need an A</a></li>
            </ul>
        </div>
    `;
}

// Expose the getFallbackContent function through the API
mapsApi.getFallbackContent = function(location) {
    return getFallbackContent(location);
};

// Export the mapsApi object
window.mapsApi = mapsApi;

// Function to try multiple search methods for clinics
function tryMultipleSearchMethods(placesService, location, map, formattedAddress) {
    console.log("Trying multiple search methods for clinics at:", formattedAddress);
    
    // Set up markers array
    const allMarkers = [];
    const bounds = new google.maps.LatLngBounds();
    bounds.extend(location);
    
    // Track successful searches
    let foundResults = false;
    let searchesCompleted = 0;
    const totalSearches = 6; // Increased to 6 search methods
    
    // Function to check if all searches are completed
    function checkAllSearchesCompleted() {
        searchesCompleted++;
        console.log(`Completed ${searchesCompleted} of ${totalSearches} searches`);
        
        if (searchesCompleted === totalSearches) {
            if (foundResults) {
                console.log(`Found ${allMarkers.length} total clinics across all searches`);
                
                // Fit map to show all markers
                if (allMarkers.length > 1) {
                    map.fitBounds(bounds);
                    
                    // Don't zoom in too far
                    google.maps.event.addListenerOnce(map, 'idle', function() {
                        if (map.getZoom() > 14) {
                            map.setZoom(14);
                        }
                    });
                } else if (allMarkers.length === 1) {
                    // Just one marker, center on it
                    map.setCenter(allMarkers[0].getPosition());
                    map.setZoom(14);
                }
            } else {
                console.log("No results found from any search method, showing synthetic data");
                
                // Show a message in the map before showing synthetic data
                const infoElement = document.createElement('div');
                infoElement.style.position = 'absolute';
                infoElement.style.top = '10px';
                infoElement.style.left = '10px';
                infoElement.style.right = '10px';
                infoElement.style.backgroundColor = 'rgba(255,255,255,0.9)';
                infoElement.style.padding = '10px';
                infoElement.style.borderRadius = '5px';
                infoElement.style.fontSize = '14px';
                infoElement.style.zIndex = '1000';
                infoElement.innerHTML = `
                    <p style="margin: 0 0 8px 0;"><strong>Note:</strong> We couldn't find real clinic data through Google Maps for "${formattedAddress}".</p>
                    <p style="margin: 0;"><small>Showing estimated locations below. For accurate information, please visit <a href="https://www.plannedparenthood.org/health-center" target="_blank">Planned Parenthood</a>.</small></p>
                `;
                map.getDiv().appendChild(infoElement);
                
                // If no clinics found, use synthetic data
                displaySyntheticData(map.getDiv(), formattedAddress.split(',')[0]);
            }
        }
    }
    
    // Function to process results and add markers
    function processResults(results, searchType) {
        if (results && results.length > 0) {
            console.log(`Found ${results.length} results via ${searchType}`);
            foundResults = true;
            
            results.forEach(place => {
                if (place.geometry && place.geometry.location) {
                    // Check if this is likely a clinic based on name
                    const placeName = place.name.toLowerCase();
                    let isLikelyClinic = false;
                    
                    // Check for common health clinic terms
                    const clinicKeywords = [
                        'planned parenthood', 
                        'family planning', 
                        'women', 
                        'health', 
                        'clinic', 
                        'medical', 
                        'hospital', 
                        'center',
                        'reproductive',
                        'birth'
                    ];
                    
                    isLikelyClinic = clinicKeywords.some(keyword => placeName.includes(keyword));
                    
                    // Skip this place if it's clearly not a clinic
                    const exclusionKeywords = ['church', 'school', 'fire department', 'police', 'restaurant', 'cafe', 'store'];
                    const shouldExclude = exclusionKeywords.some(keyword => placeName.includes(keyword));
                    
                    if (shouldExclude) {
                        console.log(`Skipping result "${place.name}" as it matches exclusion keywords`);
                        return;
                    }
                    
                    // Create marker
                    const marker = new google.maps.Marker({
                        position: place.geometry.location,
                        map: map,
                        title: place.name,
                        animation: google.maps.Animation.DROP
                    });
                    
                    allMarkers.push(marker);
                    
                    // Create info window
                    const infoWindow = new google.maps.InfoWindow({
                        content: `
                            <div class="info-window">
                                <h3>${place.name}</h3>
                                <p>${place.vicinity || place.formatted_address || 'Address unavailable'}</p>
                                ${place.rating ? `<p>Rating: ${place.rating} ⭐</p>` : ''}
                                ${place.opening_hours?.open_now ? 
                                    '<p>Status: Open now</p>' : 
                                    place.opening_hours ? '<p>Status: Closed</p>' : ''}
                            </div>
                        `
                    });
                    
                    // Add click listener
                    marker.addListener('click', () => {
                        infoWindow.open(map, marker);
                    });
                    
                    // Extend bounds
                    bounds.extend(place.geometry.location);
                }
            });
        } else {
            console.log(`No results found via ${searchType}`);
        }
    }
    
    // 1. Try "planned parenthood" with nearby search
    const request1 = {
        location: location,
        radius: 30000, // 30km radius
        type: ['health'],
        keyword: 'planned parenthood'
    };
    
    placesService.nearbySearch(request1, (results, status) => {
        console.log(`Nearby Search for "planned parenthood" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'nearby search - planned parenthood');
        }
        checkAllSearchesCompleted();
    });
    
    // 2. Try "women's health" with nearby search
    const request2 = {
        location: location,
        radius: 30000,
        type: ['health'],
        keyword: "women's health"
    };
    
    placesService.nearbySearch(request2, (results, status) => {
        console.log(`Nearby Search for "women's health" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'nearby search - women\'s health');
        }
        checkAllSearchesCompleted();
    });
    
    // 3. Try "reproductive health" with nearby search
    const request3 = {
        location: location,
        radius: 30000,
        type: ['health'],
        keyword: "reproductive health"
    };
    
    placesService.nearbySearch(request3, (results, status) => {
        console.log(`Nearby Search for "reproductive health" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'nearby search - reproductive health');
        }
        checkAllSearchesCompleted();
    });
    
    // 4. Try text search for "planned parenthood"
    const request4 = {
        query: `planned parenthood near ${formattedAddress}`,
        location: location,
        radius: 30000
    };
    
    placesService.textSearch(request4, (results, status) => {
        console.log(`Text Search for "planned parenthood" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'text search - planned parenthood');
        }
        checkAllSearchesCompleted();
    });
    
    // 5. Try text search for "women's health clinic"
    const request5 = {
        query: `women's health clinic near ${formattedAddress}`,
        location: location,
        radius: 30000
    };
    
    placesService.textSearch(request5, (results, status) => {
        console.log(`Text Search for "women's health clinic" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'text search - women\'s health clinic');
        }
        checkAllSearchesCompleted();
    });
    
    // 6. Try Find Place for "planned parenthood"
    const request6 = {
        fields: ['name', 'geometry', 'formatted_address', 'place_id'],
        locationBias: {
            center: location,
            radius: 30000
        },
        input: 'planned parenthood',
        inputType: 'textQuery'
    };
    
    placesService.findPlaceFromQuery(request6, (results, status) => {
        console.log(`Find Place for "planned parenthood" status: ${status}`);
        if (status === google.maps.places.PlacesServiceStatus.OK) {
            processResults(results, 'find place - planned parenthood');
        }
        checkAllSearchesCompleted();
    });
}

// Function to display synthetic data
function displaySyntheticData(mapElement, location) {
    console.log("Displaying synthetic data for", location);
    
    // Get synthetic clinic data
    const clinics = getClinicDataForLocation(location);
    
    // Create HTML for synthetic data
    const html = `
        <div style="padding: 15px;">
            <h3>Abortion Clinics Near ${location}</h3>
            <p><small>Note: Showing approximate locations based on your area</small></p>
            ${clinics.map(clinic => `
                <div style="margin: 15px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <strong>${clinic.name}</strong><br>
                    ${clinic.address}<br>
                    <span style="color: #666;">Phone: ${clinic.phone}</span>
                </div>
            `).join('')}
            <p>For accurate clinic information, please visit: <a href="https://www.plannedparenthood.org/health-center" target="_blank">Planned Parenthood Health Center Finder</a></p>
        </div>
    `;
    
    // Display in map element
    if (typeof mapElement === 'string') {
        const element = document.getElementById(mapElement);
        if (element) element.innerHTML = html;
    } else if (mapElement instanceof HTMLElement) {
        mapElement.innerHTML = html;
    }
}