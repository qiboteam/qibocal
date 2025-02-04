function redirectToRuncard() {
    window.location.href = './action.yml';
}

function redirectToPlatform() {
    window.location.href = './new_platform/parameters.json';
}


// To Download PDF
var exportPDFButton = document.getElementById("export-pdf");

exportPDFButton.addEventListener("click", function() {

document.body.classList.add("impresion");

var doc = new jsPDF({orientation: 'landscape',});

var iframes = document.querySelectorAll("iframe.gh-fit")
source = ""
for(var id = 0; id < iframes.length; id++) {
    var win = iframes[id].contentWindow
    var doc = win.document
    var html = doc.documentElement
    var body = doc.body
    var ifrm = iframes[id] // or win.frameElement
    source = source + html
}

print(source)

doc.fromHTML(source, 0, 0, {
width: 210,
margins: {
    left: 10,
    right: 10,
    top: 10,
    bottom: 10
}
});
doc.save("report.pdf");
});
