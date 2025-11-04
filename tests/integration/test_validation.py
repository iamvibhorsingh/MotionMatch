#!/usr/bin/env python3
"""Comprehensive validation script for MotionMatch MVP test setup"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestValidator:
    """Validates the MotionMatch test setup and functionality"""
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.session = requests.Session()
        self.session.timeout = 30
    
    def validate_api_health(self) -> Dict[str, Any]:
        """Validate API health and readiness"""
        logger.info("Validating API health...")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            # Health check
            response = self.session.get(f"{self.api_base}/health")
            if response.status_code == 200:
                health = response.json()
                result["details"]["health"] = health
                
                if health.get("status") == "healthy":
                    result["status"] = "healthy"
                    logger.info("✓ API is healthy")
                else:
                    result["status"] = "unhealthy"
                    result["errors"].append(f"API status: {health.get('status')}")
            else:
                result["status"] = "error"
                result["errors"].append(f"Health check failed: HTTP {response.status_code}")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Health check error: {str(e)}")
        
        return result
    
    def validate_system_stats(self) -> Dict[str, Any]:
        """Validate system statistics"""
        logger.info("Validating system statistics...")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            response = self.session.get(f"{self.api_base}/stats")
            if response.status_code == 200:
                stats = response.json()
                result["details"]["stats"] = stats
                
                # Validate expected fields
                required_fields = ["total_videos", "model_name", "device", "vector_dim"]
                missing_fields = [field for field in required_fields if field not in stats]
                
                if missing_fields:
                    result["status"] = "incomplete"
                    result["errors"].append(f"Missing stats fields: {missing_fields}")
                else:
                    result["status"] = "complete"
                    logger.info(f"✓ System stats complete: {stats['total_videos']} videos indexed")
            else:
                result["status"] = "error"
                result["errors"].append(f"Stats request failed: HTTP {response.status_code}")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Stats error: {str(e)}")
        
        return result
    
    def validate_search_functionality(self, test_video_path: Path) -> Dict[str, Any]:
        """Validate search functionality with a test video"""
        logger.info(f"Validating search with: {test_video_path.name}")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            with open(test_video_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'top_k': 5,
                    'enable_reranking': False
                }
                
                response = self.session.post(
                    f"{self.api_base}/search/upload",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                search_result = response.json()
                result["details"]["search_result"] = search_result
                
                # Validate search result structure
                required_fields = ["query_id", "processing_time_ms", "results", "total_results"]
                missing_fields = [field for field in required_fields if field not in search_result]
                
                if missing_fields:
                    result["status"] = "incomplete"
                    result["errors"].append(f"Missing search result fields: {missing_fields}")
                else:
                    processing_time = search_result.get("processing_time_ms", 0)
                    num_results = len(search_result.get("results", []))
                    
                    result["status"] = "success"
                    result["details"]["processing_time_ms"] = processing_time
                    result["details"]["num_results"] = num_results
                    
                    logger.info(f"✓ Search successful: {num_results} results in {processing_time:.0f}ms")
                    
                    # Validate result structure
                    if search_result["results"]:
                        first_result = search_result["results"][0]
                        result_fields = ["video_id", "similarity_score", "distance", "video_path"]
                        missing_result_fields = [field for field in result_fields if field not in first_result]
                        
                        if missing_result_fields:
                            result["errors"].append(f"Missing result fields: {missing_result_fields}")
            else:
                result["status"] = "error"
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
                result["errors"].append(f"Search failed: {error_detail}")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Search error: {str(e)}")
        
        return result
    
    def validate_indexing_functionality(self, test_video_path: Path) -> Dict[str, Any]:
        """Validate indexing functionality"""
        logger.info(f"Validating indexing with: {test_video_path.name}")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            response = self.session.post(
                f"{self.api_base}/index/single",
                params={"video_path": str(test_video_path.absolute())},
                timeout=120
            )
            
            if response.status_code == 200:
                index_result = response.json()
                result["details"]["index_result"] = index_result
                
                if index_result.get("status") == "success":
                    result["status"] = "success"
                    logger.info("✓ Indexing successful")
                else:
                    result["status"] = "failed"
                    result["errors"].append(f"Indexing failed: {index_result}")
            else:
                result["status"] = "error"
                error_detail = response.json().get("detail", f"HTTP {response.status_code}")
                result["errors"].append(f"Indexing request failed: {error_detail}")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Indexing error: {str(e)}")
        
        return result
    
    def validate_test_videos(self, test_video_dir: Path) -> Dict[str, Any]:
        """Validate test video directory and contents"""
        logger.info("Validating test videos...")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            if not test_video_dir.exists():
                result["status"] = "missing"
                result["errors"].append(f"Test video directory does not exist: {test_video_dir}")
                return result
            
            # Find video files
            video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
            video_files = []
            
            for file_path in test_video_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    video_files.append(file_path)
            
            result["details"]["video_count"] = len(video_files)
            result["details"]["video_files"] = [
                {
                    "name": v.name,
                    "size_mb": round(v.stat().st_size / (1024 * 1024), 2),
                    "path": str(v)
                } for v in video_files
            ]
            
            if len(video_files) == 0:
                result["status"] = "empty"
                result["errors"].append("No video files found in test directory")
            elif len(video_files) < 3:
                result["status"] = "insufficient"
                result["errors"].append(f"Only {len(video_files)} videos found, recommend at least 3")
            else:
                result["status"] = "sufficient"
                logger.info(f"✓ Found {len(video_files)} test videos")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Video validation error: {str(e)}")
        
        return result
    
    def validate_test_report(self, report_path: Path) -> Dict[str, Any]:
        """Validate test report if it exists"""
        logger.info("Validating test report...")
        
        result = {
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            if not report_path.exists():
                result["status"] = "missing"
                result["errors"].append("Test report not found")
                return result
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            result["details"]["report"] = report
            
            # Validate report structure
            required_sections = ["test_timestamp", "indexing_results", "search_results", "summary"]
            missing_sections = [section for section in required_sections if section not in report]
            
            if missing_sections:
                result["status"] = "incomplete"
                result["errors"].append(f"Missing report sections: {missing_sections}")
            else:
                result["status"] = "complete"
                
                # Extract key metrics
                summary = report.get("summary", {})
                result["details"]["indexing_success_rate"] = summary.get("indexing_success_rate", 0)
                result["details"]["search_success_rate"] = summary.get("search_success_rate", 0)
                result["details"]["avg_search_time_ms"] = summary.get("avg_search_time_ms", 0)
                
                logger.info(f"✓ Test report complete")
                logger.info(f"  Indexing success: {summary.get('indexing_success_rate', 0):.1f}%")
                logger.info(f"  Search success: {summary.get('search_success_rate', 0):.1f}%")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Report validation error: {str(e)}")
        
        return result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the test setup"""
        logger.info("Running comprehensive validation...")
        
        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "unknown",
            "validations": {},
            "summary": {},
            "recommendations": []
        }
        
        # 1. Validate API health
        validation_results["validations"]["api_health"] = self.validate_api_health()
        
        # 2. Validate system stats
        validation_results["validations"]["system_stats"] = self.validate_system_stats()
        
        # 3. Validate test videos
        test_video_dir = Path("testvideo")
        validation_results["validations"]["test_videos"] = self.validate_test_videos(test_video_dir)
        
        # 4. Validate test report
        report_path = Path("test_report.json")
        validation_results["validations"]["test_report"] = self.validate_test_report(report_path)
        
        # 5. If we have test videos, validate search functionality
        video_validation = validation_results["validations"]["test_videos"]
        if video_validation["status"] in ["sufficient", "insufficient"] and video_validation["details"].get("video_files"):
            test_video_path = Path(video_validation["details"]["video_files"][0]["path"])
            validation_results["validations"]["search_functionality"] = self.validate_search_functionality(test_video_path)
        
        # Calculate overall status
        statuses = [v["status"] for v in validation_results["validations"].values()]
        error_count = sum(1 for status in statuses if status == "error")
        success_count = sum(1 for status in statuses if status in ["success", "complete", "healthy", "sufficient"])
        
        if error_count > 0:
            validation_results["overall_status"] = "failed"
        elif success_count == len(statuses):
            validation_results["overall_status"] = "passed"
        else:
            validation_results["overall_status"] = "partial"
        
        # Generate summary
        validation_results["summary"] = {
            "total_validations": len(validation_results["validations"]),
            "passed": success_count,
            "failed": error_count,
            "partial": len(statuses) - success_count - error_count
        }
        
        # Generate recommendations
        recommendations = []
        
        if validation_results["validations"]["api_health"]["status"] != "healthy":
            recommendations.append("Start the MotionMatch API server: python start.py")
        
        if validation_results["validations"]["test_videos"]["status"] == "empty":
            recommendations.append("Add test videos to the testvideo/ directory")
        elif validation_results["validations"]["test_videos"]["status"] == "insufficient":
            recommendations.append("Add more test videos (recommend at least 3-5)")
        
        if validation_results["validations"]["test_report"]["status"] == "missing":
            recommendations.append("Run the test setup: python testsetup.py")
        
        validation_results["recommendations"] = recommendations
        
        return validation_results
    
    def print_validation_report(self, results: Dict[str, Any]):
        """Print a formatted validation report"""
        print("\n" + "="*60)
        print("🔍 MOTIONMATCH TEST VALIDATION REPORT")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        
        # Summary
        summary = results["summary"]
        print(f"\nSummary: {summary['passed']}/{summary['total_validations']} validations passed")
        
        # Individual validations
        print("\nValidation Details:")
        print("-" * 30)
        
        for name, validation in results["validations"].items():
            status = validation["status"]
            status_icon = {
                "success": "✅", "complete": "✅", "healthy": "✅", "sufficient": "✅",
                "partial": "⚠️", "incomplete": "⚠️", "insufficient": "⚠️",
                "failed": "❌", "error": "❌", "unhealthy": "❌", "missing": "❌", "empty": "❌"
            }.get(status, "❓")
            
            print(f"{status_icon} {name.replace('_', ' ').title()}: {status}")
            
            if validation.get("errors"):
                for error in validation["errors"][:2]:  # Show first 2 errors
                    print(f"    • {error}")
        
        # Recommendations
        if results["recommendations"]:
            print("\nRecommendations:")
            print("-" * 15)
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"{i}. {rec}")
        
        # Key metrics (if available)
        if "test_report" in results["validations"] and results["validations"]["test_report"]["status"] == "complete":
            details = results["validations"]["test_report"]["details"]
            print(f"\nTest Metrics:")
            print(f"  Indexing Success Rate: {details.get('indexing_success_rate', 0):.1f}%")
            print(f"  Search Success Rate: {details.get('search_success_rate', 0):.1f}%")
            print(f"  Average Search Time: {details.get('avg_search_time_ms', 0):.0f}ms")
        
        print("="*60)

def main():
    """Main function"""
    validator = TestValidator()
    
    try:
        results = validator.run_comprehensive_validation()
        validator.print_validation_report(results)
        
        # Save detailed results
        with open("validation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Detailed validation report saved to: validation_report.json")
        
        # Exit with appropriate code
        if results["overall_status"] == "passed":
            print("🎉 All validations passed!")
            sys.exit(0)
        elif results["overall_status"] == "partial":
            print("⚠️ Some validations passed with warnings")
            sys.exit(0)
        else:
            print("❌ Validation failed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()