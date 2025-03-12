const selectBox = document.querySelector('.select-contain-select');
const cardItemList = document.querySelectorAll('.card-item-list');

selectBox.addEventListener('change', function() {
    const selectedOption = this.value;
    const sortedItems = Array.from(cardItemList).sort((a, b) => {
        if (selectedOption === 'Popularity') {
            // Sort by badge popularity
            const badgeA = a.querySelector('.badge') ? a.querySelector('.badge').innerText : '';
            const badgeB = b.querySelector('.badge') ? b.querySelector('.badge').innerText : '';
            return badgeB.localeCompare(badgeA);
        } else if (selectedOption === 'low to high') {
            // Sort by price low to high
            const priceA = parseFloat(a.querySelector('.price__num').innerText.replace(/[^0-9.-]+/g,""));
            const priceB = parseFloat(b.querySelector('.price__num').innerText.replace(/[^0-9.-]+/g,""));
            return priceA - priceB;
        } else if (selectedOption === 'high to low') {
            // Sort by price high to low
            const priceA = parseFloat(a.querySelector('.price__num').innerText.replace(/[^0-9.-]+/g,""));
            const priceB = parseFloat(b.querySelector('.price__num').innerText.replace(/[^0-9.-]+/g,""));
            return priceB - priceA;
        } else if (selectedOption === 'A to Z') {
            // Sort by hotel name A to Z
            const nameA = a.querySelector('.card-title a').innerText.toLowerCase();
            const nameB = b.querySelector('.card-title a').innerText.toLowerCase();
            return nameA.localeCompare(nameB);
        }
    });

    // Append sorted items to the container
    const parentContainer = cardItemList[0].parentNode;
    parentContainer.innerHTML = ''; // Clear existing items
    sortedItems.forEach(item => parentContainer.appendChild(item));
});
