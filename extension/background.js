(function() {

  chrome.browserAction.onClicked.addListener(function(tab){
    chrome.tabs.insertCSS({file: 'style.css'});
    chrome.tabs.executeScript({file: 'jquery-3.2.1.min.js'});
    chrome.tabs.executeScript({file: 'sidebar.js'});
  });

})()