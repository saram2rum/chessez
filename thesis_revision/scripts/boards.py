import chess, chess.svg, asyncio
from playwright.async_api import async_playwright
items={
 'fig_fen_white': ('2kr2r1/pppq3p/3pbpnb/8/8/2NPPN1P/PPQB1PPK/2R3R1 w - - 15 18', None),
 'fig_mate_case': ('r6k/6nP/1pP1p1P1/3p1p1R/3q4/3P4/4KPB1/8 w - - 0 39', None),
}
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':800,'height':800}, device_scale_factor=2)
        for name,(fen,_) in items.items():
            svg=chess.svg.board(chess.Board(fen), size=760, coordinates=True, orientation=chess.WHITE,
                colors={'square light':'#eeeed2','square dark':'#769656','margin':'#ffffff','coord':'#333333'})
            await pg.set_content(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
            el=await pg.query_selector('svg'); await el.screenshot(path=f'/home/claude/w/img/{name}.png')
        await b.close()
asyncio.run(main())
