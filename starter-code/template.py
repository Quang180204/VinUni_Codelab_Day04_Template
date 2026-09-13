"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: System Prompt cấp sản xuất
# Bao gồm Persona, Core Rules, Operational Boundaries và Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: tư vấn viên sản phẩm, dịch vụ và hỗ trợ khách hàng VinFast/Vinpearl.
- Giọng điệu: chuyên nghiệp, thân thiện, súc tích và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: tra cứu danh mục xe điện VinFast và dịch vụ du lịch Vinpearl.
- submit_support_ticket: ghi nhận vấn đề và tạo phiếu hỗ trợ cho khách hàng.

## CORE RULES
1. Không bịa tên, giá, tính năng, tình trạng sản phẩm hoặc mã ticket.
2. Phải gọi search_product_catalog khi người dùng muốn tìm hoặc so sánh sản phẩm theo giá.
3. Phải gọi submit_support_ticket khi người dùng yêu cầu ghi nhận một vấn đề hỗ trợ.
4. Dùng đúng dữ liệu Observation để tạo câu trả lời; thông báo rõ nếu không có kết quả.
5. Không gọi tool khi có thể trả lời một câu hỏi FAQ chung bằng kiến thức đã được cung cấp.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ các chủ đề thuộc hệ sinh thái Vingroup, đặc biệt là VinFast và Vinpearl.
- Với yêu cầu ngoài phạm vi, lịch sự nêu giới hạn và không suy đoán.
- Không thực hiện thanh toán, đặt cọc, cam kết bảo hành hoặc thay đổi dữ liệu ngoài hai tool trên.

## OUTPUT CONTRACT
- Quy trình nội bộ tuân theo: Thought -> Action -> Observation -> Final Answer.
- Action phải có tên tool và tham số hợp lệ theo JSON Schema.
- Chỉ hiển thị Final Answer cho người dùng; không tiết lộ Thought nội bộ.
- Final Answer phải bằng tiếng Việt, nêu kết quả thực tế và bước tiếp theo phù hợp.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Mô phỏng chatbot một lượt để đối chiếu với agent dùng tool."""
        return {
            "answer": (
                "[Chatbot Baseline] Tôi đã nhận được câu hỏi: "
                f"{user_input}. Câu trả lời này chưa được kiểm chứng bằng dữ liệu công cụ."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max(0, max_iterations)
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_price(user_input: str) -> int:
        """Đổi các cách viết như '600 triệu' hoặc '1,2 tỷ' sang VNĐ."""
        match = re.search(
            r"(?:dưới|không\s+quá|tối\s+đa|đến|<=?)\s*"
            r"([\d.,]+)\s*(tỷ|tỉ|triệu|tr|nghìn|ngàn|k|đồng|vnd)?",
            user_input,
            flags=re.IGNORECASE,
        )
        if not match:
            return 999999999999

        raw_value, unit = match.groups()
        unit = (unit or "").lower()
        if unit in {"tỷ", "tỉ", "triệu", "tr"}:
            number = float(raw_value.replace(".", "").replace(",", "."))
        else:
            number = float(raw_value.replace(".", "").replace(",", ""))

        multiplier = {
            "tỷ": 1_000_000_000,
            "tỉ": 1_000_000_000,
            "triệu": 1_000_000,
            "tr": 1_000_000,
            "nghìn": 1_000,
            "ngàn": 1_000,
            "k": 1_000,
        }.get(unit, 1)
        return int(number * multiplier)

    @staticmethod
    def _extract_customer_name(user_input: str) -> str:
        patterns = [
            r"(?:tôi\s+tên|tên\s+tôi)\s+(?:là\s+)?([^,.;]+)",
            r"tôi\s+là\s+([^,.;]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.split(
                    r"\s+(?:xe|phòng|đang|bị|muốn|cần)\b",
                    name,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                if name:
                    return name
        return "Khách hàng"

    @staticmethod
    def _extract_priority(user_input: str) -> str:
        normalized = user_input.lower()
        if any(word in normalized for word in ("nghiêm trọng", "khẩn cấp", "xử lý gấp", "ưu tiên cao")):
            return "high"
        if any(word in normalized for word in ("không gấp", "ưu tiên thấp", "mức độ thấp")):
            return "low"
        return "medium"

    @staticmethod
    def _extract_issue(user_input: str, customer_name: str) -> str:
        issue = user_input.strip()
        if "phản hồi:" in issue.lower():
            issue = re.split(r"phản\s+hồi\s*:\s*", issue, maxsplit=1, flags=re.IGNORECASE)[1]

        name_patterns = [
            rf"(?:tôi\s+tên|tên\s+tôi)\s+(?:là\s+)?{re.escape(customer_name)}\s*,?\s*",
            rf"tôi\s+là\s+{re.escape(customer_name)}\s*,?\s*",
        ]
        for pattern in name_patterns:
            issue = re.sub(pattern, "", issue, count=1, flags=re.IGNORECASE)

        issue = re.sub(
            r"[.!?]?\s*(?:đây\s+là\s+)?(?:vấn\s+đề\s+)?(?:nghiêm\s+trọng|khẩn\s+cấp)"
            r"[^.!?]*[.!?]?\s*$",
            "",
            issue,
            flags=re.IGNORECASE,
        ).strip(" ,.;")
        return issue or "Yêu cầu hỗ trợ của khách hàng"

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        normalized = user_input.lower()
        catalog_markers = (
            "muốn xem", "cho tôi xem", "tìm", "giá dưới", "giá tối đa",
            "không quá", "có xe", "có resort", "sản phẩm nào",
        )
        ticket_markers = (
            "ghi nhận phản hồi", "hỗ trợ", "khiếu nại", "bị lỗi", "lỗi hệ thống",
            "ẩm mốc", "vấn đề nghiêm trọng", "xử lý gấp",
        )
        needs_catalog = any(marker in normalized for marker in catalog_markers)
        needs_ticket = any(marker in normalized for marker in ticket_markers)

        category = "du_lich" if any(
            marker in normalized for marker in ("vinpearl", "resort", "du lịch", "phòng")
        ) else "xe_dien"
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "category": category,
            "max_price": self._extract_price(user_input),
        }

    @staticmethod
    def _format_catalog(results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp."
        if results[0].get("error"):
            return "Rất tiếc, hiện không thể truy cập dữ liệu danh mục sản phẩm."

        lines = ["Các lựa chọn phù hợp:"]
        for product in results:
            price = f"{product['price_vnd']:,}".replace(",", ".")
            lines.append(f"- {product['name']}: {price} VNĐ")
        return "\n".join(lines)

    @staticmethod
    def _faq_answer(user_input: str) -> str:
        normalized = user_input.lower()
        if "bảo hành" in normalized and "pin" in normalized:
            return (
                "Theo thông tin trong danh mục của bài lab, pin xe điện VinFast được bảo hành "
                "10 năm. Điều kiện chi tiết có thể khác theo mẫu xe và thị trường."
            )
        if any(brand in normalized for brand in ("vingroup", "vinfast", "vinpearl")):
            return "Tôi đã nhận câu hỏi về dịch vụ Vingroup, nhưng chưa có dữ liệu FAQ phù hợp để trả lời chính xác."
        return "Tôi chỉ có thể hỗ trợ các nội dung thuộc hệ sinh thái Vingroup, VinFast và Vinpearl."

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        actions: List[Dict[str, Any]] = []

        if intents["needs_catalog"]:
            actions.append({
                "tool": "search_product_catalog",
                "arguments": {
                    "category": intents["category"],
                    "max_price": intents["max_price"],
                },
            })

        if intents["needs_ticket"]:
            customer_name = self._extract_customer_name(user_input)
            actions.append({
                "tool": "submit_support_ticket",
                "arguments": {
                    "customer_name": customer_name,
                    "issue_description": self._extract_issue(user_input, customer_name),
                    "priority": self._extract_priority(user_input),
                },
            })

        if not actions:
            if self.max_iterations < 1:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": 0,
                    "status": "max_iterations_reached",
                }
            answer = self._faq_answer(user_input)
            self.trace.append({
                "iteration": 1,
                "thought": "Câu hỏi FAQ hoặc ngoài phạm vi, không cần gọi tool.",
                "action": None,
                "observation": None,
                "final_answer": answer,
            })
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        observations: Dict[str, Any] = {}
        iterations = 0
        for action in actions:
            if iterations >= self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iterations,
                    "status": "max_iterations_reached",
                }

            iterations += 1
            tool_name = action["tool"]
            arguments = action["arguments"]
            try:
                observation = TOOL_MAP[tool_name](**arguments)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                observation = {"error": str(error)}

            observations[tool_name] = observation
            self.trace.append({
                "iteration": iterations,
                "thought": f"Cần dùng {tool_name} để lấy dữ liệu đã kiểm chứng.",
                "action": tool_name,
                "action_input": arguments,
                "observation": observation,
            })

        answer_parts: List[str] = []
        if "search_product_catalog" in observations:
            answer_parts.append(self._format_catalog(observations["search_product_catalog"]))
        if "submit_support_ticket" in observations:
            ticket = observations["submit_support_ticket"]
            if ticket.get("error"):
                answer_parts.append("Không thể tạo ticket hỗ trợ vào lúc này.")
            else:
                answer_parts.append(
                    f"Đã tạo ticket {ticket['ticket_id']} cho {ticket['customer_name']} "
                    f"với mức ưu tiên {ticket['priority']}."
                )

        answer = "\n\n".join(answer_parts)
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
