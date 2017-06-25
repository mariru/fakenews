(function () {

  const addArticle = (article) => {
    $('#sidebar-content').append(`
      <div class="sidebar-article">
        <div class="article-header">
          ${article.publisher}
        </div>
        <div class="article-content">
          <h1>${article.headline}</h1>
          <div>${article.summary}</div>
        </div>
      </div>
    `);
  }

  // Initialize sidebar.
  if (!$('#pop-main-container').length) {
    const $innerContent = $('body').html();
    $('body').html(`
      <div id="pop-main-container">
        ${$innerContent}
      </div>
      <div id="pop-sidebar-container">
        <div id="sidebar-header">
          <div id="pop-main-logo">
            <div id="pop-logo-text">
              Pop
            </div>
          </div>
        </div>
        <div id="sidebar-content"></div>
        <div id="sidebar-footer"></div>
      </div>
    `);
  }

})();