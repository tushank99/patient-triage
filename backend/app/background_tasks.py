import asyncio
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.app.database import SessionLocal
from backend.app.models import Visit

# Safe Wait Times by ESI Level (in seconds for the hackathon demo)
# In reality, ESI 2 is 10 mins, ESI 3 is 60 mins. 
# We use short times (e.g., 60 seconds) so the judges can actually see the deterioration trigger live.
ESI_WAIT_THRESHOLDS = {
    1: 0,       # Immediate
    2: 30,      # 30 seconds for demo
    3: 60,      # 60 seconds for demo
    4: 120,
    5: 240
}

async def deterioration_monitor():
    """
    Loops constantly in the background. If a patient waits longer than 
    their ESI threshold, it flags them for re-assessment.
    """
    print("Deterioration Monitor Started...")
    while True:
        try:
            db: Session = SessionLocal()
            # Find everyone currently waiting
            waiting_visits = db.query(Visit).filter(Visit.status == "WAITING").all()
            
            now = datetime.now(timezone.utc)
            updates_made = False
            
            for visit in waiting_visits:
                esi = visit.final_esi_score
                if not esi:  # type: ignore
                    continue
                    
                threshold = ESI_WAIT_THRESHOLDS.get(esi, 3600)  # type: ignore
                
                arrival = visit.arrival_time
                if arrival.tzinfo is None:
                    # If SQLite stripped the timezone, put it back as UTC
                    arrival = arrival.replace(tzinfo=timezone.utc)
                
                wait_time_seconds = (now - arrival).total_seconds()  # type: ignore
                
                # If they exceeded their safe wait time
                if wait_time_seconds > threshold:
                    visit.status = "NEEDS_REASSESSMENT"  # type: ignore
                    updates_made = True
            
            if updates_made:
                db.commit()
                print("⚠️ Deterioration Monitor: Flagged patients for reassessment!")
                
            db.close()
        except Exception as e:
            print(f"Monitor error: {e}")
            
        await asyncio.sleep(5)  # Check every 5 seconds for the fast-paced demo
