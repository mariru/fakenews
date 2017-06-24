(function () {

  if (!document.getElementById('pop-main-container')) {
    const mainContainer = document.createElement('div');
    mainContainer.id = 'pop-main-container';
    mainContainer.innerHTML = document.body.innerHTML;

    const sidebarContainer = document.createElement('div');
    sidebarContainer.id = 'pop-main-sidebar';
    document.body.innerHTML = mainContainer.outerHTML;
    document.body.appendChild(sidebarContainer);
  }

})();