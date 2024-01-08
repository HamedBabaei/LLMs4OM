
function fetchSelectedCSVAndRenderPivotTable(){
    if(window.location != window.parent.location)
        $("<a>", {target:"_blank", href:""})
            .text("[pop out]").prependTo($("body"));
    selectedFile = document.getElementById('csvFiles').value;
    fetch(selectedFile)
        .then(response => response.text())
        .then(csvData => {
            // Generate Pivot Table
            $("#pivotTableContainer").pivotUI(
                $.csv.toArrays(csvData),
                {
                    renderers: $.extend(
                        $.pivotUtilities.renderers,
                        $.pivotUtilities.c3_renderers,
                        $.pivotUtilities.d3_renderers,
                        $.pivotUtilities.export_renderers
                    ),
                    hiddenAttributes: [""]
                }
            ).show();
        })
        .catch(error => {
            console.error('Error fetching CSV:', error);
        });
}


function capturePivotTable() {
    // Use html2canvas to capture the content inside the pivotTableContainer
    html2canvas(document.getElementById('pivotTableContainer')).then(function(canvas) {
        // Ask the user to specify the filename
        var fileName = window.prompt('Enter file name', 'pivot_table_capture.png');

        // If the user provides a filename and doesn't cancel the prompt
        if (fileName) {
            // Convert canvas to an image data URL
            var image = canvas.toDataURL(); // This is the image data

            // Create a link (<a>) element
            var downloadLink = document.createElement('a');
            downloadLink.href = image;
            downloadLink.download = fileName; // Use the specified filename

            // Simulate a click on the download link
            downloadLink.click();
        }
    });
}

document.getElementById('csvFiles').addEventListener('change', fetchSelectedCSVAndRenderPivotTable);
document.getElementById('captureButton').addEventListener('click', capturePivotTable);

fetchSelectedCSVAndRenderPivotTable();
