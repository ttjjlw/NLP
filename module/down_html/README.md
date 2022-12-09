报错：selenium.common.exceptions.ElementClickInterceptedException: Message: element click intercepted: Element <div data-v-351caf1a="" data-v-6adcd48c="" class="select-container">...</div> is not clickable at point (223, 567). Other element would receive the click: <div class="cover-cut-footer">...</div>
  (Session info: headless chrome=108.0.5359.98)
解决办法：
    try:
        driver.find_element_by_xpath('//*[@class="submit-add" and text()="立即投稿"]').click()
    except:
        element = driver.find_element_by_xpath('//*[@class="submit-add" and text()="立即投稿"]')
        driver.execute_script("arguments[0].click();", element)

报错：no such element: Unable to locate element: {"method":"xpath","selector":"//*[@class="f-item-content" and text()="知识"]"}
解决办法：
    查看是否不同条件，显现情况不一样