// Setup canvas
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
context.strokeStyle = '#222222';
context.lineWidth = 2;

// Global state
var drawing;
var pos;
var lastPos;
var strokes;
var currentStroke;
function clearCanvas() {
    canvas.width = canvas.width;
    drawing = false;
    pos = {x: 0, y: 0};
    lastPos = pos;
    strokes = [];
    currentStroke = [];
}
clearCanvas();

// Mouse events
function getMousePos(canvas, e) {
    var rectangle = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rectangle.left,
        y: e.clientY - rectangle.top
    }
}
canvas.addEventListener('mousedown', function (e) {
    drawing = true;
    lastPos = getMousePos(canvas, e);
});
canvas.addEventListener('mouseup', function (e) {
    drawing = false;

});
canvas.addEventListener('mousemove', function (e) {
    pos = getMousePos(canvas, e);
});

// Touch events
function getTouchPos(canvas, e) {
    var rectangle = canvas.getBoundingClientRect();
    return {
        x: e['touches'][0].clientX - rectangle.left,
        y: e['touches'][0].clientY - rectangle.top
    }
}
canvas.addEventListener('touchstart', function (e) {
    var touch = e['touches'][0];
    var mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});
canvas.addEventListener("touchend", function (e) {
    canvas.dispatchEvent(new MouseEvent("mouseup", {}));
});
canvas.addEventListener("touchmove", function (e) {
    var touch = e['touches'][0];
    var mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});

// Prevent scrolling when touching the canvas
document.body.addEventListener("touchstart", function (e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
});
document.body.addEventListener("touchend", function (e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
});
document.body.addEventListener("touchmove", function (e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
});

// Get animation and draw on canvas
window.requestAnimFrame = (function (callback) {
    return window.requestAnimationFrame ||
        window.webkitRequestAnimationFrame ||
        window.mozRequestAnimationFrame ||
        window.oRequestAnimationFrame ||
        window.msRequestAnimaitonFrame ||
        function (callback) {
            window.setTimeout(callback, 1000 / 60);
        };
})();
function renderPosOnCanvas() {
    if (drawing) {
        context.moveTo(lastPos.x, lastPos.y);
        context.lineTo(pos.x, pos.y);
        context.stroke();
    }
}
function savePosToStrokes() {
    if (drawing) {
        currentStroke.push(pos);
    } else if (currentStroke.length !== 0) {
        strokes.push(currentStroke);
        submit();
        currentStroke = [];
    }
}
(function renderLoop() {
    requestAnimationFrame(renderLoop);
    renderPosOnCanvas();
    savePosToStrokes();
    lastPos = pos;
})();

// submit
function submit() {
    console.log(strokes);
    $.ajax({
        type: 'POST',
        url: '/submit',
        data: JSON.stringify({data: strokes}),
        contentType: 'application/json;charset=UTF-8',
        success: function (data) {
            $('#latex').text('$' + data['latex'] + '$');
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    });
}

// clear button
$('#clear').click(function () {
    clearCanvas()
});