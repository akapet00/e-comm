import pprint

from bs4 import BeautifulSoup

HTML_CONTENT = """
    <html><body>
    <div class="newsletter">
    <div class="product">
        <h2 class="name">智能手机 (Smartphone XYZ)</h2>
        <img src="http://example.com/image1.jpg" alt="Smartphone image"/>
        <span class="price">¥2999</span>
        <a href="http://example.com/product/123">View Product</a>
    </div>
    <div class="product">
        <h2 class="name">运动鞋 (Running Shoes ABC)</h2>
        <img src="http://example.com/image2.jpg" alt="Shoes image"/>
        <span class="price">¥499</span>
        <a href="http://example.com/product/456">View Product</a>
    </div>
    </div>
    </body></html>
    """


def extract_data(html_content: str) -> list[dict[str, str]]:
    """Return extracted products.

    Return extracted products from a simple HTML snippet representing a
    Chinese newsletter from an e-mail.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    products = []
    for div in soup.find_all("div", class_="product"):
        name = div.find("h2", class_="name").get_text()
        price = div.find("span", class_="price").get_text()
        image = div.find("img")["src"]
        url = div.find("a")["href"]
        products.append(
            {"name": name, "price": price, "image_url": image, "url": url},
        )
    return products


if __name__ == "__main__":
    products = extract_data(HTML_CONTENT)
    print(f"Input:\n{HTML_CONTENT}", end="\n\n")
    pprint.pp(products)
