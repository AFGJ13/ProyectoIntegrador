import requests
import time
import random
import sys

# ─── CONFIGURACIÓN ────────────────────────────────────
IP_RECEPTOR  = '10.12.1.66'  # <-- Cambia a la IP del PC receptor
HTTP_PORT    = 5000
INTERVALO_MS = 500               # Cada cuántos ms envía un valor (500 = 0.5s)
MODO         = 'auto'            # 'auto' = aleatorio | 'manual' = tú eliges
# ──────────────────────────────────────────────────────

URL = f'http://{IP_RECEPTOR}:{HTTP_PORT}'


def enviar_valor(valor: int) -> bool:
    """Envía un valor (1, 2 o 3) al servidor receptor. Retorna True si OK."""
    try:
        r = requests.get(URL, params={'data': str(valor)}, timeout=2)
        if r.status_code == 200:
            print(f'  ✅ Enviado: {valor}  → respuesta: {r.text.strip()}')
            return True
        else:
            print(f'  ⚠ Respuesta inesperada: {r.status_code}')
            return False
    except requests.exceptions.ConnectionError:
        print(f'  ❌ Error de conexión a {URL}  — ¿está activo el receptor?')
        return False
    except requests.exceptions.Timeout:
        print(f'  ❌ Timeout — el receptor no respondió en 2 s')
        return False


def modo_auto():
    """Simula ráfagas de lecturas como las mandaría el modelo de cámara."""
    print('\n[MODO AUTO] Enviando ráfagas de valores aleatorios 1-3')
    print('Presiona Ctrl+C para detener.\n')

    rafaga = 0
    while True:
        rafaga += 1
        n_valores = random.randint(4, 10)   # cada ráfaga envía entre 4 y 10 lecturas
        print(f'── Ráfaga #{rafaga} ({n_valores} valores) ──')

        for _ in range(n_valores):
            valor = random.randint(1, 3)
            enviar_valor(valor)
            time.sleep(INTERVALO_MS / 1000)

        pausa = random.uniform(2.5, 5.0)
        print(f'  … pausa de {pausa:.1f} s entre ráfagas …\n')
        time.sleep(pausa)


def modo_manual():
    """Tú decides qué valor enviar en cada momento."""
    print('\n[MODO MANUAL] Escribe 1, 2 o 3 y presiona Enter para enviar.')
    print('Escribe "q" para salir.\n')

    while True:
        entrada = input('Valor a enviar (1/2/3) → ').strip().lower()
        if entrada == 'q':
            print('Saliendo...')
            break
        if entrada in ('1', '2', '3'):
            enviar_valor(int(entrada))
        else:
            print('  Valor inválido. Usa 1, 2 o 3.')


def verificar_conexion():
    """Prueba de conexión básica antes de empezar."""
    print(f'Verificando conexión con {URL} ...')
    try:
        r = requests.get(URL, params={'data': '1'}, timeout=3)
        if r.status_code == 200:
            print('✅ Conexión exitosa — el receptor está respondiendo.\n')
            return True
    except Exception:
        pass
    print(f'❌ No se pudo conectar a {URL}')
    print('   Verifica que:')
    print('   1. El PC receptor tiene el script principal ejecutándose')
    print('   2. La IP en este script es correcta')
    print(f'   3. El puerto {HTTP_PORT} no está bloqueado por firewall\n')
    return False


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == '__main__':

    print('=' * 50)
    print('   EMISOR DE PRUEBA — Simulador de cámara')
    print('=' * 50)

    if IP_RECEPTOR == '192.168.1.XXX':
        print('\n⚠  AVISO: Cambia IP_RECEPTOR en la parte superior del script')
        print('   con la IP real del PC que corre el receptor.\n')
        sys.exit(1)

    if not verificar_conexion():
        sys.exit(1)

    if MODO == 'manual':
        modo_manual()
    else:
        try:
            modo_auto()
        except KeyboardInterrupt:
            print('\nDetenido por el usuario.')