// Test script to check Google Maps API functionality
let mapTestStatus = {
    apiLoaded: false,
    mapCreated: false,
    placesServiceWorking: false,
    geocoderWorking: false,
    errors: []
};

// Function to initialize the map and test functionality
function initMapTest() {
    console.log("Running Google Maps API test...");
    document.getElementById('test-status').innerHTML = 'Testing Google Maps API...';
    
    try {
        // Test 1: Check if Google Maps API is loaded
        if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
            throw new Error("Google Maps API not loaded");
        }
        
        mapTestStatus.apiLoaded = true;
        logStatus("✅ Google Maps API loaded successfully");
        
        // Test 2: Try to create a map
        const testMap = new google.maps.Map(document.getElementById('test-map'), {
            center: { lat: 37.7749, lng: -122.4194 }, // San Francisco coordinates
            zoom: 12
        });
        
        if (!testMap) {
            throw new Error("Failed to create map instance");
        }
        
        mapTestStatus.mapCreated = true;
        logStatus("✅ Map instance created successfully");
        
        // Test 3: Try to use the Geocoder service
        const geocoder = new google.maps.Geocoder();
        geocoder.geocode({ 'address': 'San Francisco, CA' }, function(results, status) {
            if (status === 'OK' && results && results.length > 0) {
                mapTestStatus.geocoderWorking = true;
                logStatus("✅ Geocoder service working correctly");
                
                // Add a marker at the geocoded location
                new google.maps.Marker({
                    map: testMap,
                    position: results[0].geometry.location,
                    title: "San Francisco, CA"
                });
                
                // Center map on the geocoded location
                testMap.setCenter(results[0].geometry.location);
            } else {
                mapTestStatus.errors.push(`Geocoder failed with status: ${status}`);
                logStatus(`❌ Geocoder service failed with status: ${status}`);
            }
            
            updateFinalStatus();
        });
        
        // Test 4: Try to use the Places service
        const placesService = new google.maps.places.PlacesService(testMap);
        const request = {
            location: { lat: 37.7749, lng: -122.4194 },
            radius: '5000',
            type: ['health']
        };
        
        placesService.nearbySearch(request, function(results, status) {
            if (status === google.maps.places.PlacesServiceStatus.OK && results && results.length > 0) {
                mapTestStatus.placesServiceWorking = true;
                logStatus(`✅ Places service working correctly (found ${results.length} places)`);
                
                // Add markers for the first 5 places
                for (let i = 0; i < Math.min(5, results.length); i++) {
                    new google.maps.Marker({
                        map: testMap,
                        position: results[i].geometry.location,
                        title: results[i].name,
                        icon: {
                            url: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png'
                        }
                    });
                }
            } else {
                mapTestStatus.errors.push(`Places service failed with status: ${status}`);
                logStatus(`❌ Places service failed with status: ${status}`);
            }
            
            updateFinalStatus();
        });
        
    } catch (error) {
        mapTestStatus.errors.push(error.message);
        logStatus(`❌ Error: ${error.message}`);
        updateFinalStatus();
    }
}

// Helper function to log status updates
function logStatus(message) {
    console.log(message);
    const statusDiv = document.getElementById('test-log');
    if (statusDiv) {
        statusDiv.innerHTML += `<div>${message}</div>`;
    }
}

// Helper function to update the final status display
function updateFinalStatus() {
    // Check if all async tests have completed
    if ((mapTestStatus.geocoderWorking || mapTestStatus.errors.includes('Geocoder failed')) && 
        (mapTestStatus.placesServiceWorking || mapTestStatus.errors.includes('Places service failed'))) {
        
        const statusDiv = document.getElementById('test-status');
        if (statusDiv) {
            if (mapTestStatus.apiLoaded && mapTestStatus.mapCreated && 
                mapTestStatus.geocoderWorking && mapTestStatus.placesServiceWorking) {
                statusDiv.innerHTML = '✅ All tests passed! Google Maps API is working correctly.';
                statusDiv.className = 'success';
            } else {
                statusDiv.innerHTML = '❌ Some tests failed. See the log for details.';
                statusDiv.className = 'error';
            }
        }
    }
}

// Export the test function for global access
window.runGoogleMapsTest = function() {
    // Reset status
    mapTestStatus = {
        apiLoaded: false,
        mapCreated: false,
        placesServiceWorking: false,
        geocoderWorking: false,
        errors: []
    };
    
    const testLog = document.getElementById('test-log');
    if (testLog) {
        testLog.innerHTML = '';
    }
    
    // Load Google Maps API dynamically
    if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
        logStatus("Loading Google Maps API...");
        
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${window.googleMapsApiKey}&libraries=places,geometry&callback=initMapTest`;
        script.async = true;
        script.defer = true;
        script.onerror = function() {
            logStatus("❌ Failed to load Google Maps API script");
            const statusDiv = document.getElementById('test-status');
            if (statusDiv) {
                statusDiv.innerHTML = '❌ Failed to load Google Maps API script. Check the API key and network connection.';
                statusDiv.className = 'error';
            }
        };
        
        document.head.appendChild(script);
    } else {
        // API already loaded, run the test
        initMapTest();
    }
}; 