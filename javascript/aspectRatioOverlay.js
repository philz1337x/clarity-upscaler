let currentWidth = null;
let currentHeight = null;
let arFrameTimeout = null;

// Handle changes in dimensions and update preview rectangle accordingly
function dimensionChange(event, isWidth, isHeight) {
    // Update current dimensions
    if (isWidth) currentWidth = parseFloat(event.target.value);
    if (isHeight) currentHeight = parseFloat(event.target.value);

    // Check if in the "img2img" tab
    const inImg2img = gradioApp().querySelector("#tab_img2img")?.style.display === "block";
    if (!inImg2img || !currentWidth || !currentHeight) return;

    const targetElement = getImageTargetElement();
    if (!targetElement) return;

    const arPreviewRect = getOrCreatePreviewRect();
    updatePreviewRectangle(targetElement, arPreviewRect);
}

// Get target element based on current tab index
function getImageTargetElement() {
    const tabIndex = get_tab_index('mode_img2img');
    const selectors = [
        '#img2img_image div[data-testid=image] img',         // img2img
        '#img2img_sketch div[data-testid=image] img',        // Sketch
        '#img2maskimg div[data-testid=image] img',           // Inpaint
        '#inpaint_sketch div[data-testid=image] img'         // Inpaint sketch
    ];
    return gradioApp().querySelector(selectors[tabIndex]);
}

// Get or create the AR preview rectangle
function getOrCreatePreviewRect() {
    let arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (!arPreviewRect) {
        arPreviewRect = document.createElement('div');
        arPreviewRect.id = "imageARPreview";
        gradioApp().appendChild(arPreviewRect);
    }
    return arPreviewRect;
}

// Update position and size of the AR preview rectangle
function updatePreviewRectangle(targetElement, arPreviewRect) {
    const { top, left, width, height } = targetElement.getBoundingClientRect();
    const viewportScale = Math.min(width / targetElement.naturalWidth, height / targetElement.naturalHeight);

    // Calculate AR rectangle dimensions and position
    const arScale = Math.min((targetElement.naturalWidth * viewportScale) / currentWidth, (targetElement.naturalHeight * viewportScale) / currentHeight);
    const arRectWidth = currentWidth * arScale;
    const arRectHeight = currentHeight * arScale;
    const arRectTop = top + window.scrollY + (height / 2) - (arRectHeight / 2);
    const arRectLeft = left + window.scrollX + (width / 2) - (arRectWidth / 2);

    // Apply styles to preview rectangle
    Object.assign(arPreviewRect.style, {
        display: 'block',
        top: `${arRectTop}px`,
        left: `${arRectLeft}px`,
        width: `${arRectWidth}px`,
        height: `${arRectHeight}px`
    });

    // Hide the preview rectangle after a delay
    clearTimeout(arFrameTimeout);
    arFrameTimeout = setTimeout(() => arPreviewRect.style.display = 'none', 2000);
}

// Initialize input watchers after UI updates
function initializeInputWatchers() {
    const arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (arPreviewRect) arPreviewRect.style.display = 'none';

    const tabImg2img = gradioApp().querySelector("#tab_img2img");
    if (tabImg2img && tabImg2img.style.display === "block") {
        const inputs = gradioApp().querySelectorAll('input');
        inputs.forEach(input => {
            const isWidth = input.parentElement.id === "img2img_width";
            const isHeight = input.parentElement.id === "img2img_height";

            if ((isWidth || isHeight) && !input.classList.contains('scrollwatch')) {
                input.addEventListener('input', e => dimensionChange(e, isWidth, isHeight));
                input.classList.add('scrollwatch');
            }

            if (isWidth) currentWidth = parseFloat(input.value);
            if (isHeight) currentHeight = parseFloat(input.value);
        });
    }
}

// Call `initializeInputWatchers` after UI updates
onAfterUiUpdate(initializeInputWatchers);
