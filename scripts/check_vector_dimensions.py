import json
import numpy as np

from src.database.connection import SessionLocal
from src.database.models import Sample, VectorizedSpectrum

session = SessionLocal()

samples = session.query(Sample).all()

for sample in samples:

    spectrum = (
        session.query(VectorizedSpectrum)
        .filter_by(sample_id=sample.id)
        .first()
    )

    vector = np.asarray(spectrum.vector)

    print(
        sample.id,
        sample.number_Sample,
        len(vector),
        np.linalg.norm(vector)
    )

session.close()