
// #region Resize video transcript container
var video = document.getElementById("video");

resize_separator = document.getElementById("video_transcript_grid_separator");
resize_separator.addEventListener("mousedown", resize_mD);

function resize_mD(event) {
    if (event.button != 0) return;

    video.style.pointerEvents = "none";
    document.addEventListener("mousemove", resize_mM);
    document.addEventListener("mouseup", resize_end);
}

function resize_mM(event) {
    //1fr 8px 400px
    let node = resize_separator.parentElement;
    document.getElementById("transcript_table").hidden = true;

    var template_column = node.style.gridTemplateColumns.split(" ");
    let new_pos = (node.clientWidth - event.clientX);

    template_column[2] = new_pos + "px";
    node.style.gridTemplateColumns = template_column.join(" ");

    resize_event_subtitle()
}

function resize_end() {
    document.getElementById("transcript_table").hidden = false;
    video.style.pointerEvents = "auto";

    document.removeEventListener("mouseup", resize_end);
    document.removeEventListener("mousemove", resize_mM);
}
// #endregion

// #region Drag to scroll transcription
transcript_container = document.getElementById("transcript")
transcript_container.addEventListener("mousedown", scroll_mD);

var pos = { top: 0, left: 0, x: 0, y: 0 };
function scroll_mD(event) {
    if (event.button != 0) return;

    pos = {
        left: transcript_container.scrollLeft,
        top: transcript_container.scrollTop,
        x: event.clientX,
        y: event.clientY,
    };

    document.addEventListener('mousemove', scroll_mM);
    document.addEventListener('mouseup', scroll_end);
}

function scroll_mM(event) {
    transcript_container.style.cursor = 'grabbing';
    transcript_container.style.userSelect = 'none';

    const dx = event.clientX - pos.x;
    const dy = event.clientY - pos.y;

    transcript_container.scrollTop = pos.top - dy;
    transcript_container.scrollLeft = pos.left - dx;
}

function scroll_end() {
    transcript_container.style.cursor = 'default';
    document.removeEventListener('mouseup', scroll_end);
    document.removeEventListener('mousemove', scroll_mM);
};
// #endregion

// #region Drag to move subtitle
var subtitle_japanese = document.getElementById("subtitle_japanese");
var subtitle_english = document.getElementById("subtitle_english");
subtitle_japanese.addEventListener("mousedown", drag_mD);
subtitle_english.addEventListener("mousedown", drag_mD);

var drag_bottom = 0;
var drag_mouseY = 0;
var event_target = null;
function drag_mD(event) {
    if(event.button != 1) return;
    if(event.target.parentElement == null || event.target.parentElement.className != "word_container") return;
    event_target = event.target.parentElement.parentElement;
    if(event_target == null || event_target.className != "subtitle") return;

    video.style.pointerEvents = "none";
    drag_bottom = event_target.style.bottom.replace("px", "");
    drag_mouseY = event.clientY;

    document.addEventListener('mousemove', drag_mM);
    document.addEventListener('mouseup', drag_end);
}

function drag_mM(event) {
    const dy = event.clientY - drag_mouseY;
    let video_subtitle_container = document.getElementById("video_subtitle_container");
    const maxBottom = video_subtitle_container.clientHeight;

    event_target.style.bottom = Math.min(Math.max(drag_bottom - dy, -event_target.clientHeight), maxBottom) + "px";
}

function drag_end() {
    video.style.pointerEvents = "auto";
    document.removeEventListener('mouseup', drag_end);
    document.removeEventListener('mousemove', drag_mM);
}

window.addEventListener("resize", resize_event_subtitle);
function resize_event_subtitle() {
    let video_subtitle_container = document.getElementById("video_subtitle_container");
    const maxBottom = video_subtitle_container.clientHeight;

    subtitle_japanese.style.bottom = Math.min(Math.max(subtitle_japanese.style.bottom.replace("px", ""), -subtitle_japanese.clientHeight), maxBottom) + "px";
    subtitle_japanese.style.left = 0;
    subtitle_japanese.style.left = (subtitle_japanese.parentElement.offsetWidth - subtitle_japanese.offsetWidth) / 2 + "px";

    subtitle_english.style.bottom = Math.min(Math.max(subtitle_english.style.bottom.replace("px", ""), -subtitle_english.clientHeight), maxBottom) + "px";
    subtitle_english.style.left = 0;
    subtitle_english.style.left = (subtitle_english.parentElement.offsetWidth - subtitle_english.offsetWidth) / 2 + "px";
}
// #endregion

document.getElementById("btn_load").addEventListener("click", load_video);
initialize();

function initialize() {
    document.getElementById("txt_url").value = localStorage.getItem("url");
    if (localStorage.getItem("url") != null) {
        load_video();
    }
}

let transcript_table = document.getElementById("transcript_table");
transcript_table.addEventListener("dblclick", function (event) {
    let tr_element = event.target.parentElement;
    if(tr_element.tagName != "TR") return;

    tr_element.className = "tr_selected";
    if(current_id != -1)
        document.getElementById(current_id).className = "tr_unselected";
    current_id = tr_element.id;

    let time = tr_element.children[0].innerText;
    let text = tr_element.children[1].innerHTML;
    let string_split = ["", text];

    // if <br> is in the text, split it
    if(text.includes("<br>")) {
        text = text.split("<br>");
        if(text.length > 1) {
            string_split[0] = text[0];
            string_split[1] = text[1];
        }
    }

    let english = string_split[0];
    let japanese = string_split[1];

    process_text(japanese, english);

    var audio = new Audio('/audio/' + time.replace(/:/g, "") + '.wav');
    audio.play();
});

var player = null;
function load_youtube(id) {
    if(player != null) {
        player.loadVideoById(id, 0);
        return;
    }

    video_id = id;
    var tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    var firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
}

function onYouTubeIframeAPIReady() {
    player = new YT.Player('video', {
        videoId: video_id,
        events: {
            //'onReady': function (e) {e.target.playVideo();}
        }
    });
}

function load_video() {
    let video_url = document.getElementById("txt_url").value;
    localStorage.setItem("url", video_url);

    load_youtube(video_url.split("v=")[1])

    let form = new FormData();
    form.append("text", video_url);
    fetch("/transcribe", {
        method: "POST",
        body: form
    });
}

var socket = io();
socket.on('update', function(data) {
    console.log("Got some data.");
    if (data.time != null) {
        last_time = data.time;
        data.english = data.translate;
        data.japanese = data.transcribe;
        if (data.english != "" || data.japanese != "") {
            append_transcription(data);
        }
    }
});

var transcription = [];
var current_id = -1;
function append_transcription(data) {
    // data => {time: "00:00:00", english: "Hello", japanese: "こんにちは"}
    transcription.push(data);

    let transcript_table = document.getElementById("transcript_table");
    let eng_html = `${data.english}<br>`
    if (data.english == "") eng_html = "";
    transcript_table.innerHTML = `<tr id=${transcription.length} class="tr_unselected"><td style="text-align:center">${data.time}</td><td>${eng_html}${data.japanese}</td></tr>` + transcript_table.innerHTML;
    if(current_id == -1 || current_id == transcription.length - 1) {
        if(current_id != -1)
            document.getElementById(current_id).className = "tr_unselected";

        current_id = transcription.length;
        document.getElementById(current_id).className = "tr_selected";

        process_text(data.japanese, data.english);
    }
}

var text_cache = {};
function process_text(japanese, english) {
    if (japanese != null && japanese != "") {
        subtitle_japanese.setAttribute("current", japanese);
        subtitle_english.setAttribute("current", english);
        if (text_cache[japanese] != null) {
            load_text_to_subtitle(text_cache[japanese], english);
        } else {
            load_text_to_subtitle(japanese, english);

            let form = new FormData();
            form.append("text", japanese);

            fetch("/process", {
                method: "POST",
                body: form
            }).then(response => response.json()).then(data => {
                if (data.error == null) {
                    text_cache[japanese] = data;
                    if (subtitle_japanese.getAttribute("current") == japanese) {
                        load_text_to_subtitle(data, english);
                    }
                }
            });
        }
    }
}

function load_text_to_subtitle(data, text="") {
    let innerHTML = "";
    // if data is a string
    if (typeof data == "string") {
        innerHTML =
        `<div class="word_container">
            <span class="word" index=${0}>${data}</span>
        </div>`;
    } else {
        for(var i = 0; i < data.length; i++) {
            innerHTML +=
            `<div class="word_container">
                <span class="word" index=${i}>${data[i]['text']}</span>
                <div class="info" state="reading_form">
                    ${data[i]['reading']}
                </div>
            </div>`;
        }
    }

    subtitle_japanese.innerHTML = innerHTML;
    if(text != "") {
        subtitle_english.innerHTML =
        `<div class="word_container">
            <span class="word" index=${0}>${text}</span>
        </div>`;
    } else {
        subtitle_english.innerHTML = "";
    }

    resize_event_subtitle()
    subtitle_japanese.querySelectorAll(".info").forEach(function (element) {
        element.parentElement.addEventListener("mousedown", set_dictionary);
    });
}

var last_word = null;
var last_info = null;
function set_reading_form(event) {
    let word_array = text_cache[subtitle_japanese.getAttribute("current")];
    if(word_array == null) return;

    let index = parseInt(last_word.getAttribute("index"));
    if (index < 0 || index >= word_array.length || word_array[index] == null) return;

    let text = word_array[index]["reading"];
    if(text == null || text == "")
        text = "No reading form";

    last_info.innerHTML = text;
    document.removeEventListener("mouseup", set_reading_form);
}

function set_dictionary(event) {
    if(event.button != 0) return;

    let word_element = event.target.parentElement.children[0];
    let info_element = event.target.parentElement.children[1];

    let word_array = text_cache[subtitle_japanese.getAttribute("current")];
    if(word_array == null) return;

    let index = parseInt(word_element.getAttribute("index"));
    if (index < 0 || index >= word_array.length || word_array[index] == null) return;

    let text = word_array[index]["text"];
    if (text == null || text == "")
        text = "No dictionary form";
    else
        text = word_array[index]["definitions"].join('<br>');

    info_element.innerHTML = text;

    last_word = word_element;
    last_info = info_element;
    document.addEventListener("mouseup", set_reading_form);
}



