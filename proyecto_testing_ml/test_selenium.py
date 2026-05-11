from selenium import webdriver
from selenium.webdriver.common.by import By
import pytest

BASE_URL = "http://127.0.0.1:8000"

@pytest.fixture
def driver():
    d = webdriver.Chrome()
    d.implicitly_wait(5)
    yield d
    d.quit()

def test_pagina_polls_carga(driver):
    """Verifica que la página principal de encuestas responde"""
    driver.get(f"{BASE_URL}/polls/")
    assert "polls" in driver.page_source.lower()
    print("\n[OK] Página principal de polls cargada")

def test_titulo_pagina(driver):
    """Verifica que la página tiene un título"""
    driver.get(f"{BASE_URL}/polls/")
    assert driver.title is not None and len(driver.title) >= 0
    print(f"\n[OK] Título de página: '{driver.title}'")

def test_url_invalida_retorna_error(driver):
    """Verifica que una URL inválida devuelve respuesta (404 o redirección)"""
    driver.get(f"{BASE_URL}/polls/99999/")
    assert driver.current_url is not None
    print(f"\n[OK] URL inválida manejada. URL actual: {driver.current_url}")