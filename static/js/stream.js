$(document).ready(function() {
// on page load, add the loading class to the body to display loading page
  $("body").addClass("loading");


//refreshes the page every 15 minutes to clear the screen (just in case of spam tweets or words get too big)
  setTimeout(function(){
  	window.location.reload(1);
  }, 60000 * 15);

});

//https://stackoverflow.com/questions/23066488/json-passed-from-python-flask-into-javascript-is-displaying-on-screen/23143651#23143651
//fetches the data from flask
function getStream() {
	var deferredData = new jQuery.Deferred();

	$.ajax({
		type: "GET",
		url: "/stream",
		success: function(data) {
			deferredData.resolve(data);
		},

		complete: function(textStatus) {
			if (textStatus === "error" || textStatus === "parseerror") {
				if ($("body").hasClass("loading")) {
					$("#fetching").text("This is taking longer than usual...");
					$("#fetching").css({fontSize: "2.5vw"});
				} else {
					$("body").addClass("error");
				}

			}
		}
	});

	return deferredData;
}

//stores the data from the ajax request
var dataDict;

//When data comes through, converts the JSON object to a JS object and processes data.
function sortData() {
	var dataDeferred = getStream();
	$.when( dataDeferred ).done( function(data) {
		var JSONdict = JSON.stringify(data);
		dataDict = JSON.parse(JSONdict);
		if (Object.keys(dataDict).length >= 1) {
			// removes the loading class from the body after we get our first non-empty data object
			if ($("body").hasClass("loading")) {
				$("body").removeClass("loading");
			}

			//removes the error message after a successful ajax request
			if ($("body").hasClass("error")) {
				$("body").removeClass("error");
			}

			processData();
		}




	});
}


//stores the words on the screen
var onscreen = [];

//object that will store the charts so that they can be updated
var charts = {};

//creates a div element for each of the items in the onscreen array
//and adds the element to the window (edits the element if the element was already on the screen)
function processData() {
	var fontSizeFactor = 8;

	for(var key in dataDict) {

		//calculate the font size to make sure it's not over 150
		//the 0th index of a dataDict key is the frequency of the word
		var fontSize = fontSizeFactor * dataDict[key][0];

		//max size will be 150
		if (fontSize > 150) {
			fontSize = 150;
		}

		//create a copy because we might have to delete keys during the loop
		var copyOnscreen = onscreen.slice();
		if (!(copyOnscreen.includes(key))) {
			var elem = document.createElement("div");
			elem.textContent = key;
			elem.style.color = getRandomColor();
			elem.style.fontSize = fontSize + "px";
			elem.id = key;
			elem.style.position = "absolute";
			setPosition(elem);
			elem.style.display = "none";
			document.body.appendChild(elem);

			createChart(elem);

			$("#"+key).addClass("data");
			$("#" + key).fadeIn(1500);

			onscreen.push(key);
			addEventHandlers(elem);


		} else {

			$("#"+key).animate({fontSize: fontSize + "px"});
			//the 1st index of a dataDict key is the polarity count array
			updateChart(charts[key], dataDict[key][1]);
		}

		checkCollision($("#"+key)[0]);
	}

}

function addEventHandlers(elem) {


		$(".data").on("mouseenter", function() {
			$("#"+this.id+"Chart").css("visibility", "visible");
			$("#"+this.id+"Chart").css("z-index", "100");
			$("#"+this.id+"Chart canvas").width(200);
			$("#"+this.id).css("visibility", "hidden");

		});

		$(".chart").on("mouseleave",  function() {
			$("#"+this.id).css("visibility", "hidden");
			$("#"+this.id.substr(0, this.id.length - 5)).css("visibility", "visible");

		});


		$("#"+elem.id+"Chart").on("click", function() {
			window.open("https://twitter.com/search?q=" + this.id.substr(0, this.id.length - 5) + "&src=typd");

		});

}

function updateChart(chart, data) {
	chart.data.datasets[0].data = data;
	chart.update();
}

function createChart(elem) {
	var canvasDiv = document.createElement("div");
	canvasDiv.id = elem.id + "Chart";
	canvasDiv.className = "chart";
	canvasDiv.style.position = "absolute";
	canvasDiv.style.left = parseFloat(window.getComputedStyle(elem).left);
	canvasDiv.style.top = parseFloat(window.getComputedStyle(elem).top);
	canvasDiv.style.visibility = "hidden";
	document.body.appendChild(canvasDiv);

	$("#"+canvasDiv.id).append("<canvas></canvas>");

	var ctx = $("#"+canvasDiv.id + " canvas");

	var pieChart = new Chart(ctx, {
		type: "pie",
		data: {
			labels: ["positive tweets", "neutral tweets", "negative tweets"],
			datasets: [{
				label: "# of tweets",
				backgroundColor: ["#6ECF9F", "#F1E554", "#B04848"],
				data: dataDict[elem.id][1]
			}]
		},
		options: {
			title: {
				display: true,
				text: elem.id
			},
			responsive: false,
			maintainAspectRatio: true,
			legend: false
		}


	});

	charts[elem.id] = pieChart;


}

//https://stackoverflow.com/questions/1484506/random-color-generator
function getRandomColor() {
	var lettersNumbers = "0123456789ABCDEF";
	var color = "#";
	for (var i = 0; i < 6; i++) {
		color += lettersNumbers[Math.floor(Math.random() * 16)];
	}

	return color;
}


/* Since the font size increases for all elements, this function checks if two elements collide on the screen.
* If elements collide, removes the element with the smaller font size and deletes that element from onscreen and data
* and deletes its chart from the charts object.
*/
function checkCollision(elem) {
	var rect1 = elem.getBoundingClientRect();
	var elemFontSize = parseFloat($("#"+elem.id).css("font-size"));

	for(var i = 0; i < onscreen.length; i++) {
		var otherElem = $("#"+onscreen[i])[0];
		if (otherElem != elem) {
			var rect2 = otherElem.getBoundingClientRect();
			var otherElemFontSize = parseFloat($("#"+otherElem.id).css("font-size"));
			if (!(rect1.right < rect2.left || rect1.left > rect2.right || rect1.bottom < rect2.top || rect1.top > rect2.bottom)) {
				if(elemFontSize >= otherElemFontSize) {

					//the key == the element's id
					$("#"+otherElem.id).fadeOut(300, function(){ $(this).remove(); });
					$("#"+otherElem.id+"Chart").remove();
					onscreen.splice(i, 1);

					delete dataDict[otherElem.id];
				} else {
					$("#"+elem.id).fadeOut(300, function(){ $(this).remove(); });
					$("#"+elem.id+"Chart").remove();

					var index = onscreen.indexOf(elem.id);
					if (index > -1) {
						onscreen.splice(index, 1);
					}

					delete dataDict[elem.id];
				}
			}


		}

	}

}


//sets the position of the new element
//does its best to make sure that the element doesn't collide with another on the screen
function setPosition(elem) {
	var loops = 0;
	var overlap;

	while (true) {

		//get random left and top position
		var randLeft = Math.round(Math.random()  * (window.innerWidth - 200 - 25 + 1) + 25);
		var randTop = Math.round(Math.random() * (window.innerHeight - 100 - 25 + 1) + 25);

		//an object with left, top, right, bottom properties of the element using randLeft and randTop
		var position = {
			left: randLeft,
			top: randTop,
			right: randLeft + 250, //300
			bottom: randTop + 150 //200
		};

		//just in case the loop doesn't exit, we count the number of iterations
		//if looped under 5 times, use larger padding to separate elements
		if (loops < 5) {
			overlap = isOverlapping(elem, position, 100, 200);
		//if looped over 5 times but under 10, use smaller padding
		} else if (loops < 10) {
			overlap = isOverlapping(elem, position, 50, 100);
		//if looped over 10 times, place it randomly anywhere
		} else {
			overlap = false;
		}

		if(overlap == false) {
			elem.style.left = position.left + "px";
			elem.style.top = position.top + "px";
			break;
		} else {
			loops++;
		}

	}
}

//checks if the random positions calculated in setPosition will cause overlapping between elements
function isOverlapping(elem, position, paddingV, paddingH) {
	var isOverlapping = false;

	for(var i = 0; i < onscreen.length; i++) {
		var otherElem = $("#"+onscreen[i])[0];
		if (otherElem != elem) {
			var elemLeft = position.left;
			var elemTop = position.top;
			var elemBottom = position.bottom;
			var elemRight = position.right;
			var otherElemLeft = parseFloat(window.getComputedStyle(otherElem).left);
			var otherElemTop = parseFloat(window.getComputedStyle(otherElem).top);
			var otherElemBottom = otherElemTop + $("#"+otherElem.id).height();
			var otherElemRight = otherElemLeft + $("#"+otherElem.id).width();


			if(elemBottom < otherElemTop - paddingV ||
			   elemTop > otherElemBottom + paddingV ||
			   elemRight < otherElemLeft - paddingH ||
			   elemLeft > otherElemRight + paddingH) {

				isOverlapping = false;

			} else {

				return true;
			}

		}
	}

	return isOverlapping;

}


sortData();

//data is fetched and sorted every 30 seconds
var timer = setInterval(function() {
	sortData();
}, 30000);
