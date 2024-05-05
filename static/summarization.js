function copySummary() {
    /* Get the summarized text */
    var summaryText = document.getElementById("summaryText").innerText;

    /* Create a textarea element */
    var textarea = document.createElement("textarea");

    /* Set the value of the textarea to the summarized text */
    textarea.value = summaryText;

    /* Append the textarea to the document body */
    document.body.appendChild(textarea);

    /* Select the text inside the textarea */
    textarea.select();

    /* Copy the selected text */
    document.execCommand("copy");

    /* Remove the textarea from the document body */
    document.body.removeChild(textarea);
};
