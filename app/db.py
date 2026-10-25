from datetime import datetime, timezone
from sqlalchemy import create_engine, Integer, Float, String, DateTime, select, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

class Base(DeclarativeBase):
    pass

class Prediction(Base):
    """One row per classified upload: what was decided, when, by which model, how confidently."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(20))
    p_ai: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20))
    mode: Mapped[str] = mapped_column(String(10))           # local | remote (cloud API)
    latency_ms: Mapped[float] = mapped_column(Float)

class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)               # fails loudly at start-up if the DB is unreachable
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def add(self, **fields):
        with self.Session.begin() as s:
            row = Prediction(**fields)
            s.add(row)
        return row

    def get(self, pid):
        with self.Session() as s:
            return s.get(Prediction, pid)

    def page(self, page, per_page=20):
        with self.Session() as s:
            total = s.scalar(select(func.count(Prediction.id)))
            rows = s.scalars(select(Prediction).order_by(Prediction.id.desc())
                             .offset((page - 1) * per_page).limit(per_page)).all()
        return rows, total

    def ping(self):
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))
