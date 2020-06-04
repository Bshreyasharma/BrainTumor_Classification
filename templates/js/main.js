var SubmitButton = document.getElementById("submit-button");
var load_container = document.getElementById("load_container");
SubmitButton.addEventListener("click", function(){
    console.log("It's working perfectly fine")
});

function renderHTML(data){
    load_container.insertAdjacentHTML('beforeend','testing 123');
    }
