from urllib import parse

def clean_url(url):
    # Remove query params from url.
    p = parse.urlparse(url)
    return parse.urlunparse((p.scheme, p.netloc, p.path, '', '', ''))
