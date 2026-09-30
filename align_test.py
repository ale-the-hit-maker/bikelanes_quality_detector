"""
Allinea i dati di Sensor Logger a un sistema di riferimento in cui la gravità
media punta sempre lungo -y (telefono "virtualmente" perfettamente verticale).

Uso:
    python align_to_gravity.py Gravity.csv Accelerometer.csv Gyroscope.csv ...
(il primo file deve essere il Gravity.csv della STESSA registrazione;
 gli altri sensori vengono ruotati con la stessa matrice R)
"""
import numpy as np
import pandas as pd

G = 9.80665
TARGET = np.array([0.0, -1.0, 0.0])   # in questi dati la gravità è ~ (0, -9.8, 0)


def median_gravity_direction(grav_xyz: np.ndarray) -> np.ndarray:
    """Direzione media (versore) della gravità nel riferimento del telefono."""
    g = np.median(grav_xyz, axis=0)          # mediana: robusta ad urti/buche
    return g / np.linalg.norm(g)


def rotation_align(u: np.ndarray, t: np.ndarray = TARGET) -> np.ndarray:
    """Rotazione minima (Rodrigues) R tale che R @ u = t."""
    v = np.cross(u, t) # create third vector perpendicular to gravity vector and the target (0, -1, 0). This is the vector around which we'll rotate
    c = float(np.dot(u, t)) # cos of angle between the vector u and t
    s = np.linalg.norm(v) # sin of the same angle
    if s < 1e-12: # ?
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], # anti-symmetric matrix associated with the x product
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]
                ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2) # create the rotation matrix with rodrigues formula


def rotation_about_x(u: np.ndarray) -> np.ndarray:
    """Variante vincolata: solo rotazione attorno a x (asse del manubrio)."""
    phi = np.pi - np.arctan2(u[2], u[1])
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def apply_rotation(df: pd.DataFrame, R: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out[["x", "y", "z"]] = df[["x", "y", "z"]].to_numpy() @ R.T   # v' = R v (per righe)
    return out


if __name__ == "__main__":
    path = "/home/alessandro/Desktop/UNI/erasmus/courses/dalm/bikelanes_quality_detector/raw_data/bumpy_stadio/Gravity.csv"
    grav = pd.read_csv(path)
    u = median_gravity_direction(grav[["x", "y", "z"]].to_numpy())
    R = rotation_align(u)          # oppure rotation_about_x(u)

    tilt_deg = np.degrees(np.arccos(np.clip(u @ TARGET, -1, 1)))
    print("direzione gravità (x,y,z):", u.round(4), "-> inclinazione:", round(tilt_deg, 2), "gradi")
    print("R =\n", R.round(5))



    # actually align
    df = pd.read_csv(path)
    out = apply_rotation(df, R)
    out_path = path.replace(".csv", "_aligned.csv")
    out.to_csv(out_path, index=False)
    print("salvato", out_path)