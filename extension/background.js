(function() {

  chrome.browserAction.onClicked.addListener((tab) => {
    chrome.tabs.insertCSS({file: 'style.css'});
    chrome.tabs.executeScript({file: 'jquery-3.2.1.min.js'});
    chrome.tabs.executeScript({file: 'sidebar.js'});
  });

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.action) {
      case 'ADD_ARTICLE':
        const tab = sender.tab;
        sendResponse({
          title: tab.title,
          icon: tab.favIconUrl,
          url: tab.url
        });
        break;
    }
  });

})()