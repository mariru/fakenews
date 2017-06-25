(function () {

  const renderArticle = (url) => {
    $('#sidebar-content').append(`
      <div class="sidebar-article">
        <div class="article-header">
          Huffington Post
        </div>
        <div class="article-content">
          <h1>This is a Cool Headline!</h1>
          <div>${url}</div>
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

    $.get('http://localhost:8000/popnews/stats', function (data) {
      const allArticles = data.articles;
      $.each(allArticles, (idx, article) => renderArticle(article.article__url));
    });
  }

  $('#pop-sidebar-container').toggleClass('active');
  $('#pop-main-container').toggleClass('active');

})();