
function fetchSelectedCSVAndRenderPivotTable(){
    if(window.location != window.parent.location)
        $("<a>", {target:"_blank", href:""})
            .text("[pop out]").prependTo($("body"));
    selectedFile = document.getElementById('csvFiles').value;
    fetch(selectedFile)
        .then(response => response.text())
        .then(csvData => {
            // Generate Pivot Table
            $("#output").pivotUI(
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

document.getElementById('csvFiles').addEventListener('change', fetchSelectedCSVAndRenderPivotTable);

fetchSelectedCSVAndRenderPivotTable();
