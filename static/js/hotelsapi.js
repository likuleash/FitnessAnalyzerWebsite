const hotelSearchBox = document.getElementById('hotel-search-box');
const searchList = document.getElementById('search-list');
const resultGrid = document.getElementById('result-grid');

// load hotels from API
async function loadHotels(searchTerm) {
    const URL = `https://example.com/api/hotels?search=${searchTerm}`;
    const res = await fetch(URL);
    const data = await res.json();
    if (data.length > 0) {
        displayHotelList(data);
    } else {
        searchList.innerHTML = "<p>No hotels found</p>";
    }
}

function findHotels() {
    let searchTerm = hotelSearchBox.value.trim();
    if (searchTerm.length > 0) {
        searchList.classList.remove('hide-search-list');
        loadHotels(searchTerm);
    } else {
        searchList.classList.add('hide-search-list');
    }
}

function displayHotelList(hotels) {
    searchList.innerHTML = "";
    hotels.forEach(hotel => {
        let hotelListItem = document.createElement('div');
        hotelListItem.classList.add('search-list-item');
        hotelListItem.innerHTML = `
        <div class="hotel-info">
            <h3>${hotel.name}</h3>
            <p>${hotel.address}</p>
            <p>Rating: ${hotel.rating}</p>
            <p>Price: ${hotel.price}</p>
        </div>
        `;
        searchList.appendChild(hotelListItem);
        hotelListItem.addEventListener('click', async () => {
            searchList.classList.add('hide-search-list');
            hotelSearchBox.value = "";
            // Load hotel details and display
            const hotelDetails = await fetchHotelDetails(hotel.id);
            displayHotelDetails(hotelDetails);
        });
    });
}

async function fetchHotelDetails(hotelId) {
    const URL = `https://travelpartner.googleapis.com`;
    const res = await fetch(URL);
    const hotelDetails = await res.json();
    return hotelDetails;
}

function displayHotelDetails(details) {
    resultGrid.innerHTML = `
    <div class="hotel-details">
        <h3>${details.name}</h3>
        <p>${details.address}</p>
        <p>Rating: ${details.rating}</p>
        <p>Price: ${details.price}</p>
        <p>Description: ${details.description}</p>
        <!-- Add more details here as needed -->
    </div>
    `;
}

window.addEventListener('click', (event) => {
    if (event.target !== hotelSearchBox) {
        searchList.classList.add('hide-search-list');
    }
});
