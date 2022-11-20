$(document).ready(function(){
    $("#sendSub").click(function(){
        var isEmpty = $("#submitText").val();
        if(isEmpty != ""){
            $.ajax({
                url: '/',
                data: {'csrfmiddlewaretoken': '{{ csrf_token }}'}
                success: function (data) {
                    $('#getText').val(data);  # the data returned from your view
                }
          });
        }
    });
}):